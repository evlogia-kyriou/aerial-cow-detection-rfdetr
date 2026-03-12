import json
import os

dataset_path = r"D:\Demo\RF-DETR\Cow-1"
test_images_dir = os.path.join(dataset_path, "test", "images")
json_path = os.path.join(dataset_path, "test", "annotations", "_annotations.coco.json")

with open(json_path) as f:
    data = json.load(f)

# Get filenames from disk and JSON
disk_files = set(os.listdir(test_images_dir))
json_files = set(img["file_name"] for img in data["images"])

print(f"Images on disk:  {len(disk_files)}")
print(f"Images in JSON:  {len(json_files)}")
print(f"Matched:         {len(disk_files & json_files)}")
print(f"In JSON but missing on disk: {len(json_files - disk_files)}")
print(f"On disk but missing in JSON: {len(disk_files - json_files)}")

# Preview a few filenames from each to spot format differences
print(f"\nSample from disk: {list(disk_files)[:3]}")
print(f"Sample from JSON: {list(json_files)[:3]}")