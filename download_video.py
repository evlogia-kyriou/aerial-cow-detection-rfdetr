"""
Batch video downloader for aerial cow footage
Reads URLs from urls.txt and downloads with yt-dlp
"""

import subprocess
import sys
from pathlib import Path

# ── CONFIG ───────────────────────────────────────────────────────────────────
URLS_FILE  = r"D:\Demo\RF-DETR\videos\urls.txt"
OUTPUT_DIR = r"D:\Demo\RF-DETR\videos"
FORMAT     = "bestvideo[ext=mp4][height<=1080]+bestaudio/best[ext=mp4]/best"
# ─────────────────────────────────────────────────────────────────────────────

urls_path = Path(URLS_FILE)
if not urls_path.exists():
    print(f"URLs file not found: {URLS_FILE}")
    print("Create the file and add one URL per line.")
    sys.exit(1)

with open(urls_path, "r") as f:
    urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]

if not urls:
    print("No URLs found in urls.txt")
    sys.exit(1)

print(f"Found {len(urls)} URL(s). Starting downloads...\n")

success, failed = [], []

for i, url in enumerate(urls, 1):
    print(f"[{i}/{len(urls)}] {url}")
    cmd = [
        "yt-dlp",
        "-f", FORMAT,
        "--merge-output-format", "mp4",
        "--output", str(Path(OUTPUT_DIR) / "%(title)s.%(ext)s"),
        "--no-playlist",
        "--progress",
        url
    ]
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print(f"  ✓ Done\n")
        success.append(url)
    else:
        print(f"  ✗ Failed\n")
        failed.append(url)

print("─" * 50)
print(f"Downloaded : {len(success)}/{len(urls)}")
if failed:
    print(f"Failed     : {len(failed)}")
    for u in failed:
        print(f"  - {u}")
print(f"\nVideos saved to: {OUTPUT_DIR}")