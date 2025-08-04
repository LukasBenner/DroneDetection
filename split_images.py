import os
import shutil
import json
from sklearn.model_selection import train_test_split

# Paths
root_dir = "data/short_dataset"  # Change this to your dataset root
images_dir = os.path.join(root_dir, "all_frames")  # directory with all images
full_json = os.path.join(root_dir, "annotations", "full_annotations.json")  # original COCO file

# Output directories
splits = ["train", "valid", "test"]
for split in splits:
    os.makedirs(os.path.join(root_dir, split), exist_ok=True)

def remap_category_ids(coco):
    categories = coco["categories"]
    old_to_new = {cat["id"]: i for i, cat in enumerate(categories)}
    
    # Reassign ids
    for i, cat in enumerate(categories):
        cat["id"] = i

    for ann in coco["annotations"]:
        ann["category_id"] = old_to_new[ann["category_id"]]

    return coco

# Load COCO
with open(full_json, "r", encoding="utf-8") as f:
    coco = json.load(f)
    
coco = remap_category_ids(coco)

info = coco.get("info", {})
licenses = coco.get("licenses", [])
images = coco["images"]
annotations = coco["annotations"]
categories = coco["categories"]

# Optionally filter out images with no annotations
images_with_anns = set([a["image_id"] for a in annotations])
images = [img for img in images if img["id"] in images_with_anns]

# Split 80% train, 10% val, 10% test
train_imgs, temp_imgs = train_test_split(images, train_size=0.8, random_state=42)
val_imgs, test_imgs = train_test_split(temp_imgs, train_size=0.5, random_state=42)

def filter_annotations(annotations, image_subset):
    img_ids = set([img["id"] for img in image_subset])
    return [ann for ann in annotations if ann["image_id"] in img_ids]

def save_split(images, annotations, split_name):
    split_dir = os.path.join(root_dir, split_name)
    ann_path = os.path.join(split_dir, "_annotations.coco.json")
    with open(ann_path, "w", encoding="utf-8") as out:
        json.dump({
            "info": info,
            "licenses": licenses,
            "images": images,
            "annotations": annotations,
            "categories": categories
        }, out, indent=2)

    for img in images:
        src = os.path.join(images_dir, img["file_name"])
        dst = os.path.join(split_dir, img["file_name"])
        if os.path.exists(src):
            shutil.copy2(src, dst)
        else:
            print(f"[WARN] Missing image file: {src}")

# Process each split
save_split(train_imgs, filter_annotations(annotations, train_imgs), "train")
save_split(val_imgs, filter_annotations(annotations, val_imgs), "valid")
save_split(test_imgs, filter_annotations(annotations, test_imgs), "test")

print(f"✅ Done: {len(train_imgs)} train, {len(val_imgs)} valid, {len(test_imgs)} test images.")
