from roboflow import Roboflow
from pathlib import Path

ROBOFLOW_API_KEY  = "BaeTy8ILqUbULKH4kqgf"
ROBOFLOW_WORKSPACE = "cctv-d30uc"
ROBOFLOW_PROJECT   = "aerial-cows-kt2wd-waby-utbtn"
ROBOFLOW_VERSION   = 1

# Download into the project folder
import os
os.chdir(r"D:\Demo\RF-DETR")

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
dataset = project.version(ROBOFLOW_VERSION).download("coco")

print(f"\n✓ Downloaded to: {dataset.location}")

# Show what was actually downloaded
for f in Path(dataset.location).rglob("*"):
    print(f)