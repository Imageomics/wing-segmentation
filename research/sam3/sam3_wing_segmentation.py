import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch
    import numpy as np
    from PIL import Image, ImageDraw
    from transformers import Sam3Processor, Sam3Model
    from scipy.ndimage import binary_erosion
    import os

    return (
        Image,
        ImageDraw,
        Sam3Model,
        Sam3Processor,
        binary_erosion,
        mo,
        np,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # SAM3 Butterfly Wing Segmentation

    This notebook tests SAM3 (facebook/sam3) via the HuggingFace Transformers library for the butterfly wing segmentation.

    Covers:
    1. Loading and running SAM3 on a single image
    2. Batch processing multiple calibrated images
    3. Heuristic-free detection using prompts
    """)
    return


@app.cell
def _(Sam3Model, Sam3Processor, mo, torch):
    device = torch.device("cpu")
    model_id = "facebook/sam3"

    mo.status.spinner(title="Loading model...")

    print("Loading processor...")
    processor = Sam3Processor.from_pretrained(model_id)

    print("Loading model...")
    model = Sam3Model.from_pretrained(model_id)
    model.to(device)
    model.eval()

    mo.callout(mo.md("✅ **Model loaded successfully!**"), kind="success")
    return device, model, processor


@app.cell
def _(Image, mo):
    image_path = "/Users/sahasra/sam3_research/1_MARA_D_calibrated.jpg"
    image = Image.open(image_path).convert("RGB")
    mo.image(image)
    return (image,)


@app.cell
def _(
    Image,
    ImageDraw,
    binary_erosion,
    device,
    image,
    mo,
    model,
    np,
    processor,
    torch,
):
    text_prompts = ["butterfly wing"]

    inputs = processor(image, text=text_prompts, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.pred_logits[0]
    scores = torch.sigmoid(logits)

    top4_idx = scores.topk(4).indices
    top4_scores = scores.topk(4).values.tolist()

    masks = outputs.pred_masks[0][top4_idx].numpy()

    img_w, img_h = image.size
    result = image.copy()
    draw = ImageDraw.Draw(result)

    colors = ["red", "cyan", "yellow", "purple"]
    labels = ["FW_left", "FW_right", "HW_left", "HW_right"]

    for i, (mask, score, color, label) in enumerate(zip(masks, top4_scores, colors, labels)):
        mask_img = Image.fromarray((mask > 0).astype(np.uint8) * 255)
        mask_img = mask_img.resize((img_w, img_h), Image.NEAREST)
        mask_arr = np.array(mask_img) > 0
    
        # Thicker outline using multiple erosions
        eroded = binary_erosion(binary_erosion(binary_erosion(mask_arr)))
        outline = mask_arr & ~eroded
        ys, xs = np.where(outline)
        for y, x in zip(ys, xs):
            draw.point((x, y), fill=color)
    
        # Add label at center of mask
        cy, cx = int(np.mean(ys)), int(np.mean(xs))
        draw.text((cx, cy), f"{label} ({score:.2f})", fill=color)

    mo.image(result)
    return


@app.cell
def _(
    Image,
    ImageDraw,
    binary_erosion,
    device,
    image,
    mo,
    model,
    np,
    processor,
    torch,
):
    def run_sam3_yolo_classes(img, proc, mdl, dev):
        # These are the exact classes from the YOLO detection module
        yolo_classes = [
            "right forewing",
            "left forewing", 
            "right hindwing",
            "left hindwing",
            "ruler",
            "metadata label",
            "color palette"
        ]
        colors = ["cyan", "red", "purple", "yellow", "white", "orange", "green"]
    
        result = img.copy()
        draw = ImageDraw.Draw(result)
    
        for prompt, color in zip(yolo_classes, colors):
            inputs = proc(img, text=[prompt], return_tensors="pt").to(dev)
        
            with torch.no_grad():
                outputs = mdl(**inputs)
        
            logits = outputs.pred_logits[0]
            scores = torch.sigmoid(logits)
            best_idx = scores.argmax()
            best_score = scores[best_idx].item()
        
            if best_score < 0.5:
                print(f"{prompt}: no detection (score={best_score:.2f})")
                continue
        
            print(f"{prompt}: detected (score={best_score:.2f})")
            mask = outputs.pred_masks[0][best_idx].numpy()
            img_w, img_h = img.size
            mask_img = Image.fromarray((mask > 0).astype(np.uint8) * 255)
            mask_img = mask_img.resize((img_w, img_h), Image.NEAREST)
            mask_arr = np.array(mask_img) > 0
        
            eroded = binary_erosion(binary_erosion(binary_erosion(mask_arr)))
            outline = mask_arr & ~eroded
            ys, xs = np.where(outline)
            for y, x in zip(ys, xs):
                draw.point((x, y), fill=color)
            cy, cx = int(np.mean(ys)), int(np.mean(xs))
            draw.text((cx, cy), f"{prompt} ({best_score:.2f})", fill=color)
    
        return result

    result_yolo_classes = run_sam3_yolo_classes(image, processor, model, device)
    mo.image(result_yolo_classes)
    return


if __name__ == "__main__":
    app.run()
