"""
Stage 3 — Aerial Cow Detection: Video Inference
RF-DETR (RFDETRBase) with bounding boxes, live count, FPS, and density heatmap
"""

import os
import time
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rfdetr import RFDETRBase
from collections import deque

# ── CONFIG ───────────────────────────────────────────────────────────────────
CHECKPOINT   = r"D:\Demo\RF-DETR\output\first\checkpoint_best_total.pth"
VIDEO_DIR    = r"D:\Demo\RF-DETR\videos_1"
OUTPUT_DIR   = r"D:\Demo\RF-DETR\output\video_inference_1"
CONF_THRESH  = 0.3
CLASS_NAMES  = ["cow"]
RESIZE_WIDTH = 1280        # resize frame width for inference (0 = no resize)
HEATMAP_ALPHA = 0.35       # heatmap blend strength
FPS_SMOOTHING = 30         # frames to average FPS over
SKIP_FRAMES  = 0           # process every N+1 frames (0 = every frame)
# ─────────────────────────────────────────────────────────────────────────────

# ── HELPERS ───────────────────────────────────────────────────────────────────
def conf_to_color(conf):
    if conf >= 0.75: return (34, 197, 94)
    elif conf >= 0.50: return (34, 220, 197)
    else: return (60, 80, 220)

def draw_box(img, x1, y1, x2, y2, label, conf):
    color  = conf_to_color(conf)
    lw     = max(2, int((img.shape[0] + img.shape[1]) / 1000))
    font   = cv2.FONT_HERSHEY_SIMPLEX
    fscale = max(0.4, min(0.7, img.shape[1] / 2000))
    thick  = max(1, lw - 1)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, lw)
    text = f"{label} {conf:.2f}"
    (tw, th), bl = cv2.getTextSize(text, font, fscale, thick)
    ty = max(y1 - 4, th + 4)
    cv2.rectangle(img, (x1, ty - th - bl - 2), (x1 + tw + 4, ty + 2), color, -1)
    cv2.putText(img, text, (x1 + 2, ty - bl), font, fscale, (0, 0, 0), thick, cv2.LINE_AA)

def draw_heatmap(img, pred_boxes, alpha=0.35):
    h, w = img.shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)
    for x1, y1, x2, y2 in pred_boxes:
        cx, cy   = (x1 + x2) // 2, (y1 + y2) // 2
        bw, bh   = max(1, x2 - x1), max(1, y2 - y1)
        radius   = int(max(bw, bh) * 1.5)
        x0c = max(0, cx - radius); x1c = min(w, cx + radius)
        y0c = max(0, cy - radius); y1c = min(h, cy + radius)
        cv2.circle(heat[y0c:y1c, x0c:x1c], (cx - x0c, cy - y0c), radius, 1.0, -1)
    if heat.max() > 0:
        heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=30)
        heat = heat / heat.max()
    cmap      = plt.get_cmap("jet")
    heat_rgba = (cmap(heat) * 255).astype(np.uint8)
    heat_bgr  = cv2.cvtColor(heat_rgba[:, :, :3], cv2.COLOR_RGB2BGR)
    mask      = (heat > 0.05).astype(np.uint8)
    blended   = img.copy()
    blended[mask == 1] = cv2.addWeighted(img, 1 - alpha, heat_bgr, alpha, 0)[mask == 1]
    return blended

def draw_overlay(img, count, fps, frame_no, total_frames):
    lines = [
        f"Cows    : {count}",
        f"FPS     : {fps:.1f}",
        f"Frame   : {frame_no}/{total_frames}",
    ]
    font   = cv2.FONT_HERSHEY_SIMPLEX
    fscale = max(0.55, min(0.85, img.shape[1] / 1800))
    thick  = 2
    pad    = 10
    lh     = int(cv2.getTextSize("A", font, fscale, thick)[0][1] * 1.9)
    w      = max(cv2.getTextSize(l, font, fscale, thick)[0][0] for l in lines) + pad * 2
    h      = lh * len(lines) + pad * 2
    overlay = img.copy()
    cv2.rectangle(overlay, (pad, pad), (pad + w, pad + h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
    for i, line in enumerate(lines):
        y = pad * 2 + lh * i + lh // 2
        color = (0, 255, 128) if i == 0 else (255, 255, 255)
        cv2.putText(img, line, (pad * 2, y), font, fscale, color, thick, cv2.LINE_AA)
    # color legend
    legend = [(">=0.75", (34,197,94)), (">=0.50", (34,220,197)), ("<0.50", (60,80,220))]
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

# ── COLLECT VIDEO PATHS ───────────────────────────────────────────────────────
exts = ("*.mp4", "*.MP4", "*.avi", "*.AVI", "*.mov", "*.MOV", "*.mkv")
video_paths = []
for ext in exts:
    video_paths.extend(glob.glob(str(Path(VIDEO_DIR) / ext)))
video_paths.sort()

if not video_paths:
    print(f"No videos found in: {VIDEO_DIR}")
    exit(1)

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Found {len(video_paths)} video(s).\n")

# ── VIDEO LOOP ────────────────────────────────────────────────────────────────
for video_path in video_paths:
    vname = Path(video_path).name
    print(f"Processing: {vname}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Could not open: {video_path}")
        continue

    src_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # output resolution
    if RESIZE_WIDTH > 0 and src_w > RESIZE_WIDTH:
        scale   = RESIZE_WIDTH / src_w
        out_w   = RESIZE_WIDTH
        out_h   = int(src_h * scale)
    else:
        out_w, out_h = src_w, src_h

    out_path = Path(OUTPUT_DIR) / (Path(video_path).stem + "_detected.mp4")
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(str(out_path), fourcc, src_fps, (out_w, out_h))

    fps_times   = deque(maxlen=FPS_SMOOTHING)
    frame_no    = 0
    last_result = None   # reuse last result on skipped frames

    print(f"  {src_w}x{src_h} @ {src_fps:.1f}fps | {total_fr} frames | output: {out_w}x{out_h}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_no += 1

        # resize if needed
        if out_w != src_w:
            frame = cv2.resize(frame, (out_w, out_h))

        if SKIP_FRAMES > 0 and (frame_no - 1) % (SKIP_FRAMES + 1) != 0:
            # reuse last detections for skipped frames
            if last_result is not None:
                writer.write(last_result)
            continue

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        t0 = time.perf_counter()
        detections = model.predict(img_rgb, threshold=CONF_THRESH)
        infer_t    = time.perf_counter() - t0
        fps_times.append(infer_t)
        fps = 1.0 / (sum(fps_times) / len(fps_times))

        result     = frame.copy()
        pred_boxes = []
        confs      = []

        for (x1, y1, x2, y2), conf, class_id in zip(
                detections.xyxy, detections.confidence, detections.class_id):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf     = float(conf)
            class_id = int(class_id) - 1
            label    = CLASS_NAMES[class_id] if 0 <= class_id < len(CLASS_NAMES) else "obj"
            x1, y1   = max(0, x1), max(0, y1)
            x2, y2   = min(out_w-1, x2), min(out_h-1, y2)
            pred_boxes.append([x1, y1, x2, y2])
            confs.append(conf)

        #if pred_boxes:
        #    result = draw_heatmap(result, pred_boxes, HEATMAP_ALPHA)

        for (x1, y1, x2, y2), conf, class_id in zip(
                detections.xyxy, detections.confidence, detections.class_id):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf     = float(conf)
            class_id = int(class_id) - 1
            label    = CLASS_NAMES[class_id] if 0 <= class_id < len(CLASS_NAMES) else "obj"
            x1, y1   = max(0, x1), max(0, y1)
            x2, y2   = min(out_w-1, x2), min(out_h-1, y2)
            draw_box(result, x1, y1, x2, y2, label, conf)

        draw_overlay(result, len(pred_boxes), fps, frame_no, total_fr)

        writer.write(result)
        last_result = result

        if frame_no % 30 == 0 or frame_no == 1:
            pct = frame_no / total_fr * 100 if total_fr > 0 else 0
            print(f"  [{pct:5.1f}%] frame {frame_no}/{total_fr} | cows: {len(pred_boxes):3d} | fps: {fps:.1f}")

    cap.release()
    writer.release()
    print(f"  Saved → {out_path}\n")

print("All done.")