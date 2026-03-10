import torch
import numpy as np
from PIL import Image, ImageDraw
from transformers import Sam3Processor, Sam3Model
from scipy.ndimage import binary_erosion
import os

device = torch.device("cpu")
model_id = "facebook/sam3"

print("Loading processor and model...")
processor = Sam3Processor.from_pretrained(model_id)
model = Sam3Model.from_pretrained(model_id)
model.to(device)
model.eval()

IMAGE_DIR = "/fs/ess/PAS2136/Butterfly/Datasets/Ananda_STRI_Images/23699580"
OUTPUT_DIR = "/fs/scratch/PAS2136/sahasrayikuntam/sam3_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

test_images = [
    "312_ISA2_D.jpg_calibrated",
    "309_ISA2_D.jpg_calibrated",
    "305_ISA2_D.jpg_calibrated",
    "301_CAST_D.jpg_calibrated",
]

def get_mask_center(mask):
    mask_np = mask.cpu().numpy()
    ys, xs = np.where(mask_np > 0.5)
    if len(xs) == 0:
        return 0, 0
    return float(xs.mean()), float(ys.mean())

def get_mask_area(mask):
    return float(mask.cpu().numpy().sum())

def assign_wing_labels(masks):
    areas = [get_mask_area(m) for m in masks]
    sorted_by_area = sorted(range(len(masks)), key=lambda i: areas[i], reverse=True)
    forewings = sorted_by_area[:2]
    hindwings = sorted_by_area[2:]
    labels = [""] * len(masks)
    fw_sorted = sorted(forewings, key=lambda i: get_mask_center(masks[i])[0])
    labels[fw_sorted[0]] = "FW_left"
    labels[fw_sorted[1]] = "FW_right"
    hw_sorted = sorted(hindwings, key=lambda i: get_mask_center(masks[i])[0])
    labels[hw_sorted[0]] = "HW_left"
    labels[hw_sorted[1]] = "HW_right"
    return labels

label_colors = {
    "FW_left":  (255, 0, 0),
    "FW_right": (0, 200, 200),
    "HW_left":  (180, 180, 0),
    "HW_right": (150, 0, 200),
}

def draw_outline(image, masks, labels, scores):
    result = image.convert("RGB")
    draw = ImageDraw.Draw(result)
    for i, mask in enumerate(masks):
        color = label_colors.get(labels[i], (200, 200, 200))
        mask_np = (mask.cpu().numpy() > 0.5).astype(np.uint8)
        eroded = binary_erosion(mask_np)
        outline = mask_np - eroded
        ys, xs = np.where(outline > 0)
        for x, y in zip(xs[::3], ys[::3]):
            draw.ellipse([x-2, y-2, x+2, y+2], fill=color)
        cx, cy = get_mask_center(mask)
        cx, cy = int(cx), int(cy)
        label_text = f"{labels[i]} ({scores[i]:.2f})"
        text_w = len(label_text) * 8
        draw.rectangle([cx - 4, cy - 18, cx + text_w, cy + 4], fill=(0, 0, 0))
        draw.text((cx, cy - 17), label_text, fill=color)
    return result

for fname in test_images:
    img_path = os.path.join(IMAGE_DIR, fname)
    if not os.path.exists(img_path):
        print(f"Skipping {fname} - not found")
        continue

    print(f"\nProcessing: {fname}")
    image = Image.open(img_path).convert("RGB")

    inputs = processor(images=image, text="butterfly wing", return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist()
    )[0]

    masks = results["masks"]
    scores = results["scores"]
    print(f"  Found {len(masks)} masks")

    if len(masks) < 4:
        print(f"  Warning: only found {len(masks)} masks, expected 4")

    labels = assign_wing_labels(masks)
    print(f"  Labels: {labels}")

    out_img = draw_outline(image, masks, labels, scores)
    out_path = os.path.join(OUTPUT_DIR, fname.replace(".jpg_calibrated", "_outline.jpg"))
    out_img.save(out_path)
    print(f"  Saved → {out_path}")

print("\nDone!")
