"""
Aerial Cow Detection — Gradio Demo
RF-DETR (RFDETRBase) deployed on Hugging Face Spaces
"""

import os
import time
import tempfile
import cv2
import numpy as np
import gradio as gr
from pathlib import Path
from huggingface_hub import hf_hub_download
from rfdetr import RFDETRBase

# ── CONFIG ───────────────────────────────────────────────────────────────────
HF_REPO_ID    = "evlogia-kyriou/aerial-cow-detector"   # ← change this
CKPT_FILENAME = "checkpoint_best_total.pth"
CONF_THRESH   = 0.3
CLASS_NAMES   = ["cow"]
MAX_VIDEO_SEC = 30      # cap video processing at 30 seconds
RESIZE_WIDTH  = 960     # resize input for faster CPU inference
# ─────────────────────────────────────────────────────────────────────────────

# ── LOAD MODEL (once at startup) ──────────────────────────────────────────────
print("Downloading checkpoint from HF Hub...")
ckpt_path = hf_hub_download(repo_id=HF_REPO_ID, filename=CKPT_FILENAME)
print(f"Checkpoint: {ckpt_path}")

print("Loading model...")
model = RFDETRBase(num_classes=len(CLASS_NAMES), pretrain_weights=ckpt_path)
print("Model ready.\n")

# ── HELPERS ───────────────────────────────────────────────────────────────────
def conf_to_color(conf):
    if conf >= 0.75: return (34, 197, 94)
    elif conf >= 0.50: return (34, 220, 197)
    else: return (60, 80, 220)

def draw_box(img, x1, y1, x2, y2, label, conf):
    color  = conf_to_color(conf)
    lw     = max(2, int((img.shape[0] + img.shape[1]) / 800))
    font   = cv2.FONT_HERSHEY_SIMPLEX
    fscale = max(0.45, min(0.75, img.shape[1] / 1500))
    thick  = max(1, lw - 1)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, lw)
    text = f"{label} {conf:.2f}"
    (tw, th), bl = cv2.getTextSize(text, font, fscale, thick)
    ty = max(y1 - 4, th + 4)
    cv2.rectangle(img, (x1, ty - th - bl - 2), (x1 + tw + 4, ty + 2), color, -1)
    cv2.putText(img, text, (x1 + 2, ty - bl), font, fscale, (0, 0, 0), thick, cv2.LINE_AA)

def draw_count_overlay(img, count, infer_ms=None):
    lines = [f"🐄 Cows detected: {count}"]
    if infer_ms:
        lines.append(f"⏱  {infer_ms:.0f} ms")
    font   = cv2.FONT_HERSHEY_SIMPLEX
    fscale = max(0.6, min(1.0, img.shape[1] / 1200))
    thick  = 2
    pad    = 12
    lh     = int(cv2.getTextSize("A", font, fscale, thick)[0][1] * 2.0)
    w      = max(cv2.getTextSize(l, font, fscale, thick)[0][0] for l in lines) + pad * 2
    h      = lh * len(lines) + pad * 2
    overlay = img.copy()
    cv2.rectangle(overlay, (pad, pad), (pad + w, pad + h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
    for i, line in enumerate(lines):
        y = pad * 2 + lh * i + lh // 2
        cv2.putText(img, line, (pad * 2, y), font, fscale,
                    (0, 255, 128) if i == 0 else (255, 255, 255), thick, cv2.LINE_AA)

def run_inference(bgr_frame):
    h, w = bgr_frame.shape[:2]
    if RESIZE_WIDTH > 0 and w > RESIZE_WIDTH:
        scale = RESIZE_WIDTH / w
        bgr_frame = cv2.resize(bgr_frame, (int(w * scale), int(h * scale)))

    img_rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    t0 = time.perf_counter()
    detections = model.predict(img_rgb, threshold=CONF_THRESH)
    ms = (time.perf_counter() - t0) * 1000

    result = bgr_frame.copy()
    h, w   = result.shape[:2]
    count  = 0

    for (x1, y1, x2, y2), conf, class_id in zip(
            detections.xyxy, detections.confidence, detections.class_id):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        conf     = float(conf)
        class_id = int(class_id) - 1
        label    = CLASS_NAMES[class_id] if 0 <= class_id < len(CLASS_NAMES) else "obj"
        draw_box(result, max(0,x1), max(0,y1), min(w-1,x2), min(h-1,y2), label, conf)
        count += 1

    draw_count_overlay(result, count, ms)
    return result, count, ms

# ── INFERENCE FUNCTIONS ───────────────────────────────────────────────────────
def predict_image(image):
    if image is None:
        return None, "No image provided."
    bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    result, count, ms = run_inference(bgr)
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    summary = f"**Cows detected:** {count}  |  **Inference time:** {ms:.0f} ms"
    return result_rgb, summary

def predict_video(video_path):
    if video_path is None:
        return None, "No video provided."

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Could not open video."

    fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_fr   = int(MAX_VIDEO_SEC * fps)
    process_fr = min(total_fr, max_fr)

    # determine output size
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if RESIZE_WIDTH > 0 and src_w > RESIZE_WIDTH:
        scale  = RESIZE_WIDTH / src_w
        out_w  = RESIZE_WIDTH
        out_h  = int(src_h * scale)
    else:
        out_w, out_h = src_w, src_h

    tmp_out = tempfile.NamedTemporaryFile(suffix="_detected.mp4", delete=False)
    tmp_out.close()
    fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
    writer  = cv2.VideoWriter(tmp_out.name, fourcc, fps, (out_w, out_h))

    frame_no    = 0
    total_count = 0

    while frame_no < process_fr:
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1
        result, count, _ = run_inference(frame)
        total_count += count
        writer.write(result)

    cap.release()
    writer.release()

    avg_count = total_count / max(frame_no, 1)
    summary   = (f"**Frames processed:** {frame_no}  |  "
                 f"**Avg cows/frame:** {avg_count:.1f}  |  "
                 f"**Video capped at:** {MAX_VIDEO_SEC}s")
    return tmp_out.name, summary

# ── GRADIO UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="Aerial Cow Detector", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🐄 Aerial Cow Detection
    **RF-DETR** model trained on aerial/drone imagery to detect and count cows.
    Upload an image or video and the model will draw bounding boxes and report the count.
    
    - 🟢 Confidence ≥ 0.75 &nbsp;&nbsp; 🟡 Confidence ≥ 0.50 &nbsp;&nbsp; 🔴 Confidence < 0.50
    """)

    with gr.Tabs():
        # ── IMAGE TAB ──────────────────────────────────────────────────────────
        with gr.Tab("📷 Image"):
            with gr.Row():
                with gr.Column():
                    img_input  = gr.Image(type="pil", label="Upload aerial image")
                    img_button = gr.Button("Detect Cows", variant="primary")
                with gr.Column():
                    img_output  = gr.Image(type="numpy", label="Detection result")
                    img_summary = gr.Markdown()
            img_button.click(predict_image, inputs=img_input,
                             outputs=[img_output, img_summary])

        # ── VIDEO TAB ──────────────────────────────────────────────────────────
        with gr.Tab("🎥 Video"):
            with gr.Row():
                with gr.Column():
                    vid_input  = gr.Video(label=f"Upload video (max {MAX_VIDEO_SEC}s processed)")
                    vid_button = gr.Button("Detect Cows", variant="primary")
                with gr.Column():
                    vid_output  = gr.Video(label="Detection result")
                    vid_summary = gr.Markdown()
            vid_button.click(predict_video, inputs=vid_input,
                             outputs=[vid_output, vid_summary])

    gr.Markdown("""
    ---
    Built with [RF-DETR](https://github.com/roboflow/rf-detr) · Deployed on Hugging Face Spaces
    """)

if __name__ == "__main__":
    demo.launch()