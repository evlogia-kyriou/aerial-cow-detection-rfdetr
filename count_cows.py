"""
Stage 1 — Aerial Cow Detection: Image Inference
RF-DETR (RFDETRBase) with color-graded boxes, GT comparison, and info overlay
"""

import os
import json
import time
import glob
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import numpy as np
from pathlib import Path
from rfdetr import RFDETRBase

# ── CONFIG ───────────────────────────────────────────────────────────────────
CHECKPOINT  = r"D:\Demo\RF-DETR\output\first\checkpoint_best_total.pth"
INPUT_PATH  = r"D:\Demo\RF-DETR\aerial-cows-kt2wd-waby-1\test"
COCO_JSON   = r"D:\Demo\RF-DETR\aerial-cows-kt2wd-waby-1\test\_annotations.coco.json"
OUTPUT_DIR  = r"D:\Demo\RF-DETR\output\inference_own_testset_stage2"
CONF_THRESH = 0.3
IOU_THRESH = 0.3
SHOW_IMAGE  = False
CLASS_NAMES = ["cow"]
DRAW_GT     = True   # draw ground truth boxes in blue alongside predictions
# ─────────────────────────────────────────────────────────────────────────────

# ── LOAD COCO ANNOTATIONS ─────────────────────────────────────────────────────
print("Loading COCO annotations...")
with open(COCO_JSON, "r") as f:
    coco = json.load(f)

# map filename → list of [x1, y1, x2, y2] GT boxes
gt_map = {}
id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}
for ann in coco["annotations"]:
    fname = Path(id_to_filename[ann["image_id"]]).name
    x, y, bw, bh = ann["bbox"]  # COCO bbox is [x, y, width, height]
    gt_map.setdefault(fname, []).append([int(x), int(y), int(x + bw), int(y + bh)])

print(f"  Loaded GT for {len(gt_map)} images, {len(coco['annotations'])} total annotations.\n")

# ── HELPERS ───────────────────────────────────────────────────────────────────
def conf_to_color(conf):
    """Green (high) → Yellow (mid) → Red (low) in BGR."""
    if conf >= 0.75:
        return (34, 197, 94)
    elif conf >= 0.50:
        return (34, 220, 197)
    else:
        return (60, 80, 220)

def draw_box(img, x1, y1, x2, y2, label, conf):
    color  = conf_to_color(conf)
    lw     = max(2, int((img.shape[0] + img.shape[1]) / 1000))
    font   = cv2.FONT_HERSHEY_SIMPLEX
    fscale = max(0.45, min(0.75, img.shape[1] / 2000))
    thick  = max(1, lw - 1)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, lw)
    text = f"{label} {conf:.2f}"
    (tw, th), bl = cv2.getTextSize(text, font, fscale, thick)
    ty = max(y1 - 4, th + 4)
    cv2.rectangle(img, (x1, ty - th - bl - 2), (x1 + tw + 4, ty + 2), color, -1)
    cv2.putText(img, text, (x1 + 2, ty - bl), font, fscale, (0, 0, 0), thick, cv2.LINE_AA)

def draw_gt_box(img, x1, y1, x2, y2):
    color = (220, 100, 20)  # blue BGR
    lw    = max(1, int((img.shape[0] + img.shape[1]) / 1200))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, lw)

def draw_heatmap(img, pred_boxes, alpha=0.35):
    """Semi-transparent Gaussian heatmap from box centers."""
    h, w = img.shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)
    for x1, y1, x2, y2 in pred_boxes:
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        radius = int(max(bw, bh) * 1.5)
        x0c = max(0, cx - radius); x1c = min(w, cx + radius)
        y0c = max(0, cy - radius); y1c = min(h, cy + radius)
        cv2.circle(heat[y0c:y1c, x0c:x1c],
                   (cx - x0c, cy - y0c), radius, 1.0, -1)

    if heat.max() > 0:
        heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=30)
        heat = heat / heat.max()

    cmap = plt.get_cmap("jet")
    heat_rgba = (cmap(heat) * 255).astype(np.uint8)
    heat_bgr  = cv2.cvtColor(heat_rgba[:, :, :3], cv2.COLOR_RGB2BGR)
    mask = (heat > 0.05).astype(np.uint8)
    blended = img.copy()
    blended[mask == 1] = cv2.addWeighted(img, 1 - alpha, heat_bgr, alpha, 0)[mask == 1]
    return blended

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0

def count_tp_fp_fn(pred_boxes, gt_boxes, iou_thresh=0.3):
    matched = set()
    tp = 0
    for pb in pred_boxes:
        best_iou, best_j = 0, -1
        for j, gb in enumerate(gt_boxes):
            iou = compute_iou(pb, gb)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thresh and best_j not in matched:
            tp += 1
            matched.add(best_j)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn

def draw_overlay(img, count, gt_count, tp, fp, fn, avg_conf, min_conf, max_conf, infer_ms):
    diff   = count - gt_count
    diff_s = f"+{diff}" if diff > 0 else str(diff)
    prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1     = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    lines = [
        (f"Predicted : {count}",                    "white"),
        (f"GT        : {gt_count}  (diff {diff_s})", "diff"),
        (f"TP / FP / FN : {tp} / {fp} / {fn}",     "white"),
        (f"Precision : {prec:.3f}",                  "white"),
        (f"Recall    : {rec:.3f}",                   "white"),
        (f"F1        : {f1:.3f}",                    "white"),
        (f"Avg conf  : {avg_conf:.3f}",              "white"),
        (f"Min/Max   : {min_conf:.3f} / {max_conf:.3f}", "white"),
        (f"Time      : {infer_ms:.1f} ms",           "white"),
    ]

    font   = cv2.FONT_HERSHEY_SIMPLEX
    fscale = max(0.5, min(0.8, img.shape[1] / 1800))
    thick  = 2
    pad    = 10
    lh     = int(cv2.getTextSize("A", font, fscale, thick)[0][1] * 1.9)
    w      = max(cv2.getTextSize(l, font, fscale, thick)[0][0] for l, _ in lines) + pad * 2
    h      = lh * len(lines) + pad * 2

    overlay = img.copy()
    cv2.rectangle(overlay, (pad, pad), (pad + w, pad + h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

    for i, (line, style) in enumerate(lines):
        y = pad * 2 + lh * i + lh // 2
        if style == "diff" and diff != 0:
            color = (80, 80, 255) if diff > 0 else (80, 255, 80)
        else:
            color = (255, 255, 255)
        cv2.putText(img, line, (pad * 2, y), font, fscale, color, thick, cv2.LINE_AA)

    # color legend
    legend = [(">=0.75", (34,197,94)), (">=0.50", (34,220,197)), ("<0.50", (60,80,220)), ("GT", (220,100,20))]
    lx = pad * 2
    ly = pad + h + 6
    for lbl, col in legend:
        cv2.rectangle(img, (lx, ly), (lx + 14, ly + 14), col, -1)
        cv2.putText(img, lbl, (lx + 18, ly + 12), font, fscale * 0.8, (255,255,255), 1, cv2.LINE_AA)
        lx += 95

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
print("Loading model...")
model = RFDETRBase(num_classes=len(CLASS_NAMES), pretrain_weights=CHECKPOINT)
print("Model loaded.\n")

# ── COLLECT IMAGE PATHS ───────────────────────────────────────────────────────
p = Path(INPUT_PATH)
if p.is_file():
    img_paths = [str(p)]
else:
    exts = ("*.jpg", "*.jpeg", "*.png")
    img_paths = []
    for ext in exts:
        img_paths.extend(glob.glob(str(p / ext)))
    img_paths = [p for p in img_paths if Path(p).name in gt_map]
    img_paths.sort()

if not img_paths:
    print(f"No images found in: {INPUT_PATH}")
    exit(1)

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Found {len(img_paths)} image(s). Running inference...\n")
csv_path = Path(OUTPUT_DIR) / "count_summary.csv"
csv_file = open(csv_path, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["filename", "predicted", "gt", "diff", "tp", "fp", "fn", "precision", "recall", "f1", "avg_conf", "infer_ms"])

hdr = f"  {'File':<42} {'Pred':>5} {'GT':>5} {'Diff':>5} {'TP':>4} {'FP':>4} {'FN':>4} {'Prec':>7} {'Rec':>7} {'F1':>7} {'ms':>7}"
print(hdr)
print("  " + "-" * (len(hdr) - 2))

# ── ACCUMULATORS ─────────────────────────────────────────────────────────────
all_tp = all_fp = all_fn = all_pred = all_gt = 0

# ── INFERENCE LOOP ────────────────────────────────────────────────────────────
for img_path in img_paths:
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"  Could not read: {img_path}")
        continue

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w    = img_bgr.shape[:2]
    fname   = Path(img_path).name

    t0 = time.perf_counter()
    detections = model.predict(img_rgb, threshold=CONF_THRESH)
    infer_ms = (time.perf_counter() - t0) * 1000

    result     = img_bgr.copy()
    pred_boxes = []
    confs      = []

    for (x1, y1, x2, y2), conf, class_id in zip(
            detections.xyxy, detections.confidence, detections.class_id):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        conf     = float(conf)
        class_id = int(class_id) - 1  # RF-DETR class_id is 1-indexed
        label    = CLASS_NAMES[class_id] if 0 <= class_id < len(CLASS_NAMES) else "obj"
        x1, y1   = max(0, x1), max(0, y1)
        x2, y2   = min(w-1, x2), min(h-1, y2)
        draw_box(result, x1, y1, x2, y2, label, conf)
        pred_boxes.append([x1, y1, x2, y2])
        confs.append(conf)

    if pred_boxes:
        result = draw_heatmap(result, pred_boxes)
    
    gt_boxes = gt_map.get(fname, [])
    if DRAW_GT:
        for gb in gt_boxes:
            draw_gt_box(result, *gb)

    count    = len(confs)
    gt_count = len(gt_boxes)
    avg_conf = float(np.mean(confs)) if confs else 0.0
    min_conf = float(np.min(confs))  if confs else 0.0
    max_conf = float(np.max(confs))  if confs else 0.0

    tp, fp, fn = count_tp_fp_fn(pred_boxes, gt_boxes)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    diff = count - gt_count

    draw_overlay(result, count, gt_count, tp, fp, fn, avg_conf, min_conf, max_conf, infer_ms)

    out_name = Path(OUTPUT_DIR) / (Path(img_path).stem + "_detected.jpg")
    cv2.imwrite(str(out_name), result, [cv2.IMWRITE_JPEG_QUALITY, 95])

    diff_s = f"+{diff}" if diff > 0 else str(diff)
    print(f"  {fname:<42} {count:>5} {gt_count:>5} {diff_s:>5} {tp:>4} {fp:>4} {fn:>4} {prec:>7.3f} {rec:>7.3f} {f1:>7.3f} {infer_ms:>7.1f}")

    csv_writer.writerow([fname, count, gt_count, diff, tp, fp, fn,
                         f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}",
                         f"{avg_conf:.4f}", f"{infer_ms:.1f}"])
    
    all_tp += tp; all_fp += fp; all_fn += fn
    all_pred += count; all_gt += gt_count

    if SHOW_IMAGE:
        cv2.imshow("Aerial Cow Detection", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# ── OVERALL SUMMARY ───────────────────────────────────────────────────────────
print("  " + "-" * (len(hdr) - 2))
overall_prec = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
overall_rec  = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
overall_f1   = 2 * overall_prec * overall_rec / (overall_prec + overall_rec) if (overall_prec + overall_rec) > 0 else 0.0
diff_total   = all_pred - all_gt
diff_s       = f"+{diff_total}" if diff_total > 0 else str(diff_total)

print(f"\n  {'OVERALL':<42} {all_pred:>5} {all_gt:>5} {diff_s:>5} {all_tp:>4} {all_fp:>4} {all_fn:>4} {overall_prec:>7.3f} {overall_rec:>7.3f} {overall_f1:>7.3f}")
print(f"\nDone. Results saved to: {OUTPUT_DIR}")

csv_file.close()
print(f"  CSV saved → {csv_path}")

# ── COUNT vs GT BAR CHART ─────────────────────────────────────────────────
print("\nGenerating count vs GT bar chart...")
import json as _json

fnames_short = [Path(p).stem[:25] for p in img_paths]
pred_counts  = []
gt_counts    = []

csv_file_r = open(csv_path, "r")
reader = csv.DictReader(csv_file_r)
rows = list(reader)
csv_file_r.close()

pred_counts = [int(r["predicted"]) for r in rows]
gt_counts   = [int(r["gt"])        for r in rows]
fnames_short = [r["filename"][:30] for r in rows]

fig, ax = plt.subplots(figsize=(14, max(6, len(rows) * 0.28)))
y = np.arange(len(rows))
bar_h = 0.38
ax.barh(y + bar_h/2, gt_counts,   bar_h, label="GT",        color="#2563EB", alpha=0.85)
ax.barh(y - bar_h/2, pred_counts, bar_h, label="Predicted", color="#16A34A", alpha=0.85)
ax.set_yticks(y)
ax.set_yticklabels(fnames_short, fontsize=6)
ax.set_xlabel("Cow Count")
ax.set_title(f"Predicted vs GT Count per Image\nOverall: Pred={sum(pred_counts)} | GT={sum(gt_counts)} | F1={overall_f1:.3f}")
ax.legend()
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
chart_path = Path(OUTPUT_DIR) / "count_vs_gt_chart.png"
plt.savefig(chart_path, dpi=130, bbox_inches="tight")
plt.close()
print(f"  Chart saved → {chart_path}")