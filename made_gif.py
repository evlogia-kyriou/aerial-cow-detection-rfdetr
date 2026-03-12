"""
Create a demo GIF from aerial cow detection video
Extracts a clip and converts to optimized GIF for README/portfolio
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ── CONFIG ───────────────────────────────────────────────────────────────────
INPUT_VIDEO  = r"D:\Demo\RF-DETR\output\video_inference_1\The Quiet Herd： Aerial Views of Cows in the Wild #drone.f299_detected.mp4"  # ← change
OUTPUT_GIF   = r"D:\Demo\RF-DETR\output\gif\demo_2.gif"
START_SEC    = 6       # start time in seconds
DURATION_SEC = 7       # how many seconds to capture
GIF_FPS      = 8       # lower = smaller file (6-10 is good)
GIF_WIDTH    = 720     # output width in pixels (height auto)
QUALITY      = 85      # palette quality (higher = better colors, larger file)
# ─────────────────────────────────────────────────────────────────────────────

cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    print(f"Could not open: {INPUT_VIDEO}")
    exit(1)

src_fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
src_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
src_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_fr  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

start_fr  = int(START_SEC * src_fps)
end_fr    = min(int((START_SEC + DURATION_SEC) * src_fps), total_fr)
step      = max(1, int(src_fps / GIF_FPS))   # sample every Nth frame

scale     = GIF_WIDTH / src_w
gif_h     = int(src_h * scale)

print(f"Source  : {src_w}x{src_h} @ {src_fps:.1f}fps | {total_fr} frames")
print(f"Clip    : {START_SEC}s → {START_SEC + DURATION_SEC}s (frames {start_fr}–{end_fr})")
print(f"GIF     : {GIF_WIDTH}x{gif_h} @ {GIF_FPS}fps")
print(f"Sampling: every {step} frame(s)\n")

cap.set(cv2.CAP_PROP_POS_FRAMES, start_fr)

frames = []
frame_no = start_fr

while frame_no < end_fr:
    ret, frame = cap.read()
    if not ret:
        break
    if (frame_no - start_fr) % step == 0:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img   = Image.fromarray(frame_rgb).resize(
            (GIF_WIDTH, gif_h), Image.LANCZOS)
        frames.append(pil_img)
    frame_no += 1

cap.release()

if not frames:
    print("No frames captured — check START_SEC and DURATION_SEC")
    exit(1)

print(f"Captured {len(frames)} frames. Saving GIF...")

Path(OUTPUT_GIF).parent.mkdir(parents=True, exist_ok=True)
# optimize palette per-frame for better color quality
frames[0].save(
    OUTPUT_GIF,
    save_all     = True,
    append_images= frames[1:],
    duration     = int(1000 / GIF_FPS),
    loop         = 0,
    optimize     = True,
)

size_mb = Path(OUTPUT_GIF).stat().st_size / 1024 / 1024
print(f"Saved → {OUTPUT_GIF}  ({size_mb:.1f} MB)")

if size_mb > 10:
    print(f"\nTip: GIF is large ({size_mb:.1f} MB). Try reducing DURATION_SEC or GIF_WIDTH.")