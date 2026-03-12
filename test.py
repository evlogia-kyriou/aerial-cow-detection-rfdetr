import json
with open(r"D:\Demo\RF-DETR\Cow-1\test\annotations\_annotations.coco.json") as f:
    coco = json.load(f)
print(len(coco["images"]))