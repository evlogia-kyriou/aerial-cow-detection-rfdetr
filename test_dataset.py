from roboflow import Roboflow
import os
import shutil
import yaml

# ── 1. Download from Roboflow ──────────────────────────────────────────────

rf = Roboflow(api_key="BaeTy8ILqUbULKH4kqgf")
project = rf.workspace("cctv-d30uc").project("cow-pyjdt-veytp")
version = project.version(1)
dataset = version.download("coco")


# Download in COCO format (RF-DETR uses COCO)
dataset = version.download("coco")
dataset_path = dataset.location  # e.g. /path/to/ProjectName-1

# ── 2. Flatten everything into a single "test" split ──────────────────────
test_images_dir = os.path.join(dataset_path, "test", "images")
test_annot_dir  = os.path.join(dataset_path, "test", "annotations")
os.makedirs(test_images_dir, exist_ok=True)
os.makedirs(test_annot_dir,  exist_ok=True)

# Move images from train/valid/test into test/images
for split in ["train", "valid", "test"]:
    src_images = os.path.join(dataset_path, split, "images")
    if not os.path.exists(src_images):
        continue
    for img in os.listdir(src_images):
        src = os.path.join(src_images, img)
        dst = os.path.join(test_images_dir, img)
        shutil.move(src, dst)

# ── 3. Merge COCO annotation JSONs into one test JSON ─────────────────────
import json

merged = {"images": [], "annotations": [], "categories": []}
img_id_offset  = 0
ann_id_offset  = 0
categories_set = False

for split in ["train", "valid", "test"]:
    json_path = os.path.join(dataset_path, split, "_annotations.coco.json")
    if not os.path.exists(json_path):
        continue

    with open(json_path) as f:
        data = json.load(f)

    if not categories_set:
        merged["categories"] = data["categories"]
        categories_set = True

    # Remap IDs to avoid collisions
    id_map = {}
    for img in data["images"]:
        new_id = img["id"] + img_id_offset
        id_map[img["id"]] = new_id
        img["id"] = new_id
        merged["images"].append(img)

    for ann in data["annotations"]:
        ann["id"]       += ann_id_offset
        ann["image_id"]  = id_map[ann["image_id"]]
        merged["annotations"].append(ann)

    img_id_offset += max((i["id"] for i in data["images"]), default=0) + 1
    ann_id_offset += max((a["id"] for a in data["annotations"]), default=0) + 1

merged_json_path = os.path.join(test_annot_dir, "_annotations.coco.json")
with open(merged_json_path, "w") as f:
    json.dump(merged, f)

print(f"✅ Test set ready: {len(merged['images'])} images, {len(merged['annotations'])} annotations")
print(f"   Images  → {test_images_dir}")
print(f"   JSON    → {merged_json_path}")