import os
import cv2
import numpy as np
import logging
from pathlib import Path
import torch
from wing_segmenter.constants import CLASSES
from wing_segmenter.exceptions import ImageProcessingError
from wing_segmenter.resizer import resize_image
from wing_segmenter.metadata_manager import (
    update_segmentation_info,
    add_coco_annotation,
    add_coco_image_info
)

from matplotlib import cm

def get_class_color_map():
    """
    Generates a mapping from class names to unique colors using the Viridis colormap.

    Returns:
    - class_color_map (dict): Mapping from class names to BGR color tuples.
    """
    num_classes = len(CLASSES)
    viridis = cm.get_cmap('viridis', num_classes)
    class_color_map = {}
    for class_id, class_name in CLASSES.items():
        if class_id == 0:
            # Assign black color for background (optional)
            class_color_map[class_name] = (0, 0, 0)
        else:
            # Get RGBA color from Viridis and convert to BGR for OpenCV
            rgba_color = viridis(class_id / num_classes)
            rgb_color = tuple(int(255 * c) for c in rgba_color[:3])
            bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])
            class_color_map[class_name] = bgr_color
    return class_color_map

def process_image(segmenter, image_path):
    """
    Processes a single image: loads, resizes, predicts, masks, saves results, and crops by class.
    """
    try:
        logging.debug(f"Processing image: {image_path}")

        relative_path = os.path.relpath(image_path, segmenter.dataset_path)
        relative_dir = os.path.dirname(relative_path)

        image = cv2.imread(image_path)
        if image is None:
            logging.warning(f"Failed to read image: {image_path}")
            return

        logging.debug(f"Image dimensions: {image.shape}")

        # Determine padding color
        if segmenter.resize_mode == 'pad':
            padding_color_to_use = segmenter.padding_color if segmenter.padding_color is not None else 'black'
        else:
            padding_color_to_use = None  # Not used

        # Resize the image if requested
        if segmenter.size:
            logging.debug(f"Resizing image to size: {segmenter.size} with mode: {segmenter.resize_mode}")
            working_image = resize_image(
                image,
                segmenter.size,
                segmenter.resize_mode,
                padding_color_to_use,
                segmenter.interpolation
            )
            logging.debug(f"Resized image dimensions: {working_image.shape}")

            # Save resized image
            base = Path(relative_path).with_suffix("") 
            save_path = os.path.join(segmenter.resized_dir, str(base) + ".png")

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, working_image)

            logging.debug(f"Resized image saved to '{save_path}'.")
        else:
            working_image = image  # no resizing

        working_image_rgb = cv2.cvtColor(working_image, cv2.COLOR_BGR2RGB)
        logging.debug("Converted image to RGB for YOLO and SAM processing.")

        yolo_results = segmenter.yolo_model(working_image_rgb)
        logging.debug(f"YOLO detections: {len(yolo_results)}")

        # Initialize full image foreground mask
        foreground_mask = np.zeros((working_image.shape[0], working_image.shape[1]), dtype=np.uint8)

        # To collect classes detected in this image
        classes_detected = set()

        # Initialize COCO annotations for this image
        image_id = add_coco_image_info(segmenter.coco_annotations_path, relative_path, working_image.shape)

        # Initialize visualization image if enabled
        if segmenter.visualize_segmentation:
            # Start with the original image
            viz_image = working_image.copy()

        # Step 1: Aggregate detections by class and select the highest confidence detection per class
        best_detections = {}
        for result in yolo_results:
            for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                class_id = int(cls.item())
                class_name = CLASSES.get(class_id, 'unknown')

                if class_id == 0:
                    logging.debug("Detected class is background. Skipping.")
                    continue

                confidence = conf.item()

                # If this class hasn't been seen or this detection has higher confidence, update best_detections
                if class_id not in best_detections or confidence > best_detections[class_id]['confidence']:
                    best_detections[class_id] = {
                        'box': box,
                        'cls': cls,
                        'confidence': confidence
                    }

        logging.debug(f"Best detections per class: {best_detections}")

        # Step 2: Process only the best detections
        for class_id, detection in best_detections.items():
            box = detection['box']
            cls = detection['cls']
            confidence = detection['confidence']
            class_name = CLASSES.get(class_id, 'unknown')

            # Put box on CPU before converting to NumPy
            box = box.cpu() if hasattr(box, 'cpu') else box
            bbox = box.numpy().tolist() if hasattr(box, 'numpy') else box  # Convert to numpy and list

            logging.debug(f"Processing class '{class_name}' with confidence {confidence} and bounding box {bbox}")

            # Generate mask using SAM for this detection
            mask = get_mask_SAM(
                result,
                working_image_rgb,
                segmenter.sam_processor,
                segmenter.sam_model,
                segmenter.device,
                box
            )

            if mask is not None:
                logging.debug(f"Mask generated for class {class_name} (ID: {class_id})")
                logging.debug(f"Mask dimensions: {mask.shape}")

                # Save mask (preserving directory structure)
                mask_filename = f"{Path(relative_path).stem}_cls{class_id}.png"
                mask_save_path = os.path.join(segmenter.masks_dir, relative_dir, mask_filename)
                os.makedirs(os.path.dirname(mask_save_path), exist_ok=True)
                cv2.imwrite(mask_save_path, mask)
                logging.debug(f"Mask saved to '{mask_save_path}'.")

                # Overlay mask outline and bounding box on visualization image if enabled
                if segmenter.visualize_segmentation:
                    # Get the color for this class
                    color = segmenter.class_color_map.get(class_name, (0, 255, 0))  # Default to green

                    # Find contours of the mask
                    mask_binary = (mask > 0).astype(np.uint8)
                    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Draw contours on the visualization image
                    cv2.drawContours(viz_image, contours, -1, color, 2)  # -1 means draw all contours, 2 is line thickness

                    # Retrieve image dimensions
                    image_height, image_width = working_image.shape[:2]

                    # Apply padding requested
                    padding = segmenter.bbox_padding
                    x1, y1, x2, y2 = map(int, bbox)
                    padded_x1 = max(x1 - padding, 0)
                    padded_y1 = max(y1 - padding, 0)
                    padded_x2 = min(x2 + padding, image_width)
                    padded_y2 = min(y2 + padding, image_height)

                    # Draw padded bounding box
                    cv2.rectangle(viz_image, (padded_x1, padded_y1), (padded_x2, padded_y2), color=color, thickness=2)

                    # Put class label with confidence
                    label = f"{class_name} {confidence:.2f}"
                    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    # Ensure the label box doesn't go above the image
                    y_label = max(padded_y1 - text_height - baseline, 0)
                    cv2.rectangle(viz_image, (padded_x1, y_label), (padded_x1 + text_width, y_label + text_height + baseline), color, -1)
                    cv2.putText(viz_image, label, (padded_x1, y_label + text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


                # Update classes detected
                if class_name != 'background':
                    classes_detected.add(class_name)

                    if segmenter.crop_by_class:
                        logging.debug(f"Initiating crop_and_save_by_class for class '{class_name}'.")
                        crop_and_save_by_class(segmenter, working_image, mask, relative_path, class_name, class_id)

                # Aggregate foreground masks (excluding background class 0)
                class_mask = (mask > 0).astype(np.uint8) * 255

                # Log shapes before bitwise_or
                logging.debug(f"Foreground mask shape: {foreground_mask.shape}")
                logging.debug(f"Class mask shape: {class_mask.shape}")

                # Ensure that class_mask is 2D
                if class_mask.ndim > 2:
                    # Reduce to 2D by taking the maximum across channels
                    class_mask = np.max(class_mask, axis=0)
                    logging.debug("Reduced class_mask to 2D.")

                logging.debug(f"Class mask reduced shape: {class_mask.shape}")

                # Perform bitwise_or
                foreground_mask = cv2.bitwise_or(foreground_mask, class_mask)
                logging.debug("Updated foreground_mask with the new class mask.")

                # Add COCO annotation
                add_coco_annotation(
                    segmenter.coco_annotations_path,
                    image_id,
                    class_id,
                    bbox,
                    mask
                )
            else:
                logging.warning(f"No mask generated for class '{class_name}' in image: {image_path}")

        # Update segmentation info with binary flags
        if classes_detected:
            update_segmentation_info(segmenter.segmentation_info, image_path, list(classes_detected))
            logging.debug(f"Classes detected in image '{image_path}': {list(classes_detected)}")
        else:
            update_segmentation_info(segmenter.segmentation_info, image_path, [])
            logging.debug(f"No classes detected in image '{image_path}'.")

        # Save the visualization image (preserving directory structure)
        if segmenter.visualize_segmentation:
            viz_filename = f"{Path(relative_path).stem}_viz.png"
            viz_save_path = os.path.join(segmenter.viz_dir, relative_dir, viz_filename)
            os.makedirs(os.path.dirname(viz_save_path), exist_ok=True)
            cv2.imwrite(viz_save_path, viz_image)
            logging.debug(f"Segmentation visualization saved to '{viz_save_path}'.")

        # Remove background from full image if specified
        if segmenter.remove_full_background:
            if np.any(foreground_mask):
                logging.debug(f"Removing background from full image using foreground mask for image '{image_path}'.")
                full_image_bg_removed = remove_background(working_image, foreground_mask, segmenter.background_color)

                # Prepare save path for full background removal
                base = Path(relative_path).with_suffix("")
                full_image_bg_removed_save_path = os.path.join(segmenter.full_bkgd_removed_dir, str(base) + ".png")

                os.makedirs(os.path.dirname(full_image_bg_removed_save_path), exist_ok=True)
                cv2.imwrite(full_image_bg_removed_save_path, full_image_bg_removed)

                logging.debug(f"Full background removed image saved to '{full_image_bg_removed_save_path}'.")
            else:
                logging.warning(f"No foreground detected for image: {image_path}. Full background removal skipped.")

    except Exception as e:
        logging.error(f"Error processing image {image_path}: {e}")
        raise ImageProcessingError(f"Error processing image {image_path}: {e}")

def get_mask_SAM(result, image, processor, model, device, box):
    """
    Generate a combined mask using the SAM model for a specific bounding box.

    Parameters:
    - result: YOLO prediction result.
    - image (np.array): The input image in RGB format.
    - processor (SamProcessor): SAM processor.
    - model (SamModel): SAM model.
    - device (str): 'cpu' or 'cuda'.
    - box (list): Bounding box [x1, y1, x2, y2].

    Returns:
    - mask_binary (np.array): The combined binary mask.
    """
    # Extract the bounding box
    logging.debug(f"Extracting bounding box for SAM processing: {box}")
    x1, y1, x2, y2 = box
    bbox = [x1, y1, x2, y2]

    # Prepare inputs for SAM using the processor
    logging.debug("Preparing inputs for SAM.")
    inputs = processor(
        images=[image],
        input_boxes=[[bbox]],
        return_tensors="pt"
    )
    
    # Move inputs to the device where the model is
    inputs = {k: v.to(device) for k, v in inputs.items()}

    try:
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Move outputs to CPU before NumPy conversion
        masks = processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )[0].numpy()  # Directly use numpy() as tensors are already on CPU

        logging.debug(f"Shape of masks after post_process_masks: {masks.shape}")

        if masks.shape[0] == 0:
            logging.warning("No masks generated by SAM.")
            return None

        # Convert all masks to binary and combine them into a single 2D mask
        mask_binary = (masks > 0.5).astype(np.uint8)

        if mask_binary.ndim > 2:
            # Combine all masks across masks and channels
            mask_binary = np.any(mask_binary, axis=(0, 1)).astype(np.uint8) * 255
        else:
            mask_binary = mask_binary * 255

        logging.debug(f"Combined mask shape: {mask_binary.shape}")

    except Exception as e:
        logging.error(f"Error during SAM mask creation: {e}")
        return None

    return mask_binary

def overlay_mask_on_image(image, mask):
    """
    Overlay the segmentation mask on the image for visualization.

    Parameters:
    - image (np.array): Original image in RGB format.
    - mask (np.array): Segmentation mask (binary or with class values).

    Returns:
    - overlay (np.array): Image with mask overlay.
    """
    # Ensure the mask is binary (0 or 255) for proper blending
    binary_mask = (mask > 0).astype(np.uint8) * 255

    # Create a color mask using a colormap (e.g., COLORMAP_VIRIDIS)
    color_mask = cv2.applyColorMap(binary_mask, cv2.COLORMAP_VIRIDIS)
    color_mask = cv2.cvtColor(color_mask, cv2.COLOR_BGR2RGB)

    # Normalize the mask to the range 0-1 for blending
    alpha_mask = (binary_mask / 255.0) * 0.9

    # Ensure alpha_mask has the same dimensions as the image
    alpha_mask = np.stack([alpha_mask] * 3, axis=-1)  # Convert to 3 channels

    # Blend only the masked region with the original image
    overlay = image.copy()
    overlay = (1 - alpha_mask) * overlay + alpha_mask * color_mask  # Blend image and colored mask

    # Convert the blended result back to uint8 format
    overlay = overlay.astype(np.uint8)

    return overlay

def crop_and_save_by_class(segmenter, image, mask, relative_path, class_name, class_id):
    """
    Crops the image based on class-specific masks and saves them to the crops directory.
    Optionally removes the background from the cropped images.

    Parameters:
    - segmenter (Segmenter): The Segmenter instance.
    - image (np.array): The original image.
    - mask (np.array): The binary segmentation mask for the specific class.
    - relative_path (str): The relative path of the image for maintaining directory structure.
    - class_name (str): The name of the class.
    - class_id (int): The ID of the class.
    """
    logging.debug(f"Starting crop_and_save_by_class for class '{class_name}' (ID: {class_id}) in image '{relative_path}'.")

    # Extract directory structure from relative path
    relative_dir = os.path.dirname(relative_path)

    # Use the bbox_padding from the Segmenter instance
    padding = segmenter.bbox_padding
    logging.debug(f"Using bbox_padding: {padding} pixels.")

    # Since each mask corresponds to a specific class, use the binary mask directly
    class_mask = (mask > 0).astype(np.uint8) * 255
    non_zero_pixels = np.sum(class_mask > 0)
    logging.debug(f"Number of non-zero pixels in class_mask for '{class_name}': {non_zero_pixels}")

    if non_zero_pixels == 0:
        logging.debug(f"No pixels found for class '{class_name}' in image '{relative_path}'.")
        return

    # Find contours for the class mask
    contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    logging.debug(f"Found {len(contours)} contours for class '{class_name}' in image '{relative_path}'.")

    if not contours:
        logging.debug(f"No contours found for class '{class_name}' in image '{relative_path}'.")
        return

    for c_idx, contour in enumerate(contours, 1):
        x, y, w, h = cv2.boundingRect(contour)

        # Apply dynamic padding
        x = max(x - padding, 0)
        y = max(y - padding, 0)
        w = min(w + 2 * padding, image.shape[1] - x)
        h = min(h + 2 * padding, image.shape[0] - y)

        logging.debug(f"Cropped area for class '{class_name}': x={x}, y={y}, w={w}, h={h} with padding={padding}")

        # Crop image and mask
        cropped_image = image[y:y+h, x:x+w]
        cropped_mask = class_mask[y:y+h, x:x+w]

        # Prepare filenames and paths (preserving directory structure)
        crop_filename = f"{Path(relative_path).stem}_cls{class_id}.png"
        crop_save_path = os.path.join(segmenter.crops_dir, relative_dir, crop_filename)
        os.makedirs(os.path.dirname(crop_save_path), exist_ok=True)

        # Save cropped image
        cv2.imwrite(crop_save_path, cropped_image)
        logging.debug(f"Cropped '{class_name}' saved to '{crop_save_path}'.")

        # Background removal from cropped image if specified
        if segmenter.remove_crops_background:
            logging.debug(f"Starting background removal for cropped image '{crop_save_path}'.")
            cropped_image_bg_removed = remove_background(cropped_image, cropped_mask, segmenter.background_color)
            crop_bg_removed_filename = f"{Path(relative_path).stem}_cls{class_id}_bg_removed.png"
            crop_bg_removed_save_path = os.path.join(segmenter.crops_bkgd_removed_dir, relative_dir, crop_bg_removed_filename)
            os.makedirs(os.path.dirname(crop_bg_removed_save_path), exist_ok=True)

            # Save the background-removed cropped image
            cv2.imwrite(crop_bg_removed_save_path, cropped_image_bg_removed)
            logging.debug(f"Cropped '{class_name}' with background removed saved to '{crop_bg_removed_save_path}'.")

def remove_background(image, mask, bg_color='black'):
    """
    Removes the background from an image based on the provided mask.

    Parameters:
    - image (np.array): The cropped or full image.
    - mask (np.array): The binary mask corresponding to the background.
    - bg_color (str): The background color to replace ('white' or 'black').

    Returns:
    - final_image (np.array): The image with background removed/replaced.
    """
    logging.debug(f"Starting background removal with color '{bg_color}'.")

    if bg_color is None:
        logging.warning("bg_color is None. Defaulting to 'black'.")
        bg_color = 'black'

    if bg_color.lower() == 'white':
        background = np.full(image.shape, 255, dtype=np.uint8)
    elif bg_color.lower() == 'black':
        background = np.zeros(image.shape, dtype=np.uint8)
    else:
        logging.error(f"Unsupported background color '{bg_color}'.")
        raise ValueError("Unsupported background color. Choose 'white' or 'black'.")

    if mask is not None:
        logging.debug("Mask is provided. Proceeding with background removal.")
        # Ensure mask is single-channel
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            logging.debug("Converted mask to single-channel grayscale.")

        # Corrected line: use 'mask' instead of 'masks'
        mask_binary = (mask > 0).astype(np.uint8) * 255
        non_zero_mask = np.sum(mask_binary > 0)
        logging.debug(f"Number of non-zero pixels in mask_binary: {non_zero_mask}")

        if non_zero_mask == 0:
            logging.warning("Mask has no non-zero pixels. Skipping background removal.")
            return image.copy()

        # Extract foreground using mask
        masked_foreground = cv2.bitwise_and(image, image, mask=mask_binary)
        logging.debug("Applied mask to extract foreground.")

        # Extract background area
        background_mask = cv2.bitwise_not(mask_binary)
        masked_background = cv2.bitwise_and(background, background, mask=background_mask)
        logging.debug("Prepared background based on mask.")

        # Combine foreground and new background
        final_image = cv2.add(masked_foreground, masked_background)
        logging.debug("Combined foreground and new background successfully.")
    else:
        logging.warning("No mask provided. Assuming entire image is foreground.")
        final_image = image.copy()

    logging.debug("Background removal completed.")
    return final_image
