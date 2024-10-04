import os
import cv2
import numpy as np
import logging
import torch
from wing_segmenter.constants import CLASSES
from wing_segmenter.exceptions import ImageProcessingError
from wing_segmenter.resizer import resize_image

from wing_segmenter.metadata_manager import update_segmentation_info

def process_image(segmenter, image_path):
    """
    Processes a single image: loads, resizes, predicts, masks, saves results, and crops by class.

    Parameters:
    - segmenter (Segmenter): The Segmenter instance.
    - image_path (str): Path to the image file.
    """
    try:
        relative_path = os.path.relpath(image_path, segmenter.dataset_path)
        relative_dir = os.path.dirname(relative_path)

        image = cv2.imread(image_path)
        if image is None:
            logging.warning(f"Failed to read image: {image_path}")
            return

        # Determine padding color
        if segmenter.resize_mode == 'pad':
            padding_color_to_use = segmenter.padding_color if segmenter.padding_color is not None else 'black'
        else:
            padding_color_to_use = None  # Not used

        # Resize the image if requested
        if segmenter.size:
            working_image = resize_image(
                image,
                segmenter.size,
                segmenter.resize_mode,
                padding_color_to_use,
                segmenter.interpolation
            )
            # Save resized image
            save_path = os.path.join(segmenter.resized_dir, relative_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, working_image)
        else:
            working_image = image  # no resizing

        working_image_rgb = cv2.cvtColor(working_image, cv2.COLOR_BGR2RGB)

        yolo_results = segmenter.yolo_model(working_image_rgb)

        # Initialize full image foreground mask
        foreground_mask = np.zeros((working_image.shape[0], working_image.shape[1]), dtype=np.uint8)

        # To collect classes detected in this image
        classes_detected = set()

        # Process each detection
        for result in yolo_results:
            if len(result.boxes) == 0:
                logging.warning(f"No detections for image: {image_path}")
                continue

            mask = get_mask_SAM(
                result,
                working_image_rgb,
                segmenter.sam_processor,
                segmenter.sam_model,
                segmenter.device
            )

            if mask is not None:
                # Save mask
                mask_save_path = os.path.join(segmenter.masks_dir, relative_path)
                mask_save_path = os.path.splitext(mask_save_path)[0] + '_mask.png'
                os.makedirs(os.path.dirname(mask_save_path), exist_ok=True)
                cv2.imwrite(mask_save_path, mask)

                # Save visualization if enabled
                if segmenter.visualize_segmentation:
                    viz = overlay_mask_on_image(working_image_rgb, mask)
                    viz_save_path = os.path.join(segmenter.viz_dir, relative_path)
                    viz_save_path = os.path.splitext(viz_save_path)[0] + '_viz.png'
                    os.makedirs(os.path.dirname(viz_save_path), exist_ok=True)
                    cv2.imwrite(viz_save_path, cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))

                # Extract classes present in the mask
                classes_in_mask = np.unique(mask)
                for class_id in classes_in_mask:
                    if class_id in CLASSES and class_id != 0:  # Exclude background
                        classes_detected.add(CLASSES[class_id])
                if 0 in classes_in_mask:
                    classes_detected.add('background')

                # Crop by class with optional background removal
                if segmenter.crop_by_class:
                    crop_and_save_by_class(segmenter, working_image, mask, relative_path)

                # Aggregate foreground masks (excluding background class 0)
                detected_classes = result.boxes.cls.cpu().numpy().astype(int)
                unique_classes = np.unique(detected_classes)
                unique_classes = unique_classes[unique_classes != 0]  # exclude background class

                for class_id in unique_classes:
                    class_mask = (mask == class_id).astype(np.uint8) * 255
                    foreground_mask = cv2.bitwise_or(foreground_mask, class_mask)
            else:
                logging.warning(f"No mask generated for image: {image_path}")

        # Update segmentation info with binary flags
        if classes_detected:
            update_segmentation_info(segmenter.segmentation_info, image_path, list(classes_detected))
        else:
            # If no classes detected, still log the image with all classes as 0
            update_segmentation_info(segmenter.segmentation_info, image_path, [])

        # Remove background from full image if specified
        if segmenter.remove_bg_full:
            if np.any(foreground_mask):
                full_image_bg_removed = remove_background(working_image, foreground_mask, segmenter.background_color)

                # Prepare save path for full background removal
                full_image_bg_removed_save_path = os.path.splitext(os.path.join(segmenter.full_bkgd_removed_dir, relative_path))[0] + '_bg_removed.png'
                os.makedirs(os.path.dirname(full_image_bg_removed_save_path), exist_ok=True)
                cv2.imwrite(full_image_bg_removed_save_path, full_image_bg_removed)
                logging.info(f"Full image with background removed saved to '{full_image_bg_removed_save_path}'.")
            else:
                logging.warning(f"No foreground detected for image: {image_path}. Full background removal skipped.")

    except Exception as e:
        logging.error(f"Error processing image {image_path}: {e}")
        raise ImageProcessingError(f"Error processing image {image_path}: {e}")

def get_mask_SAM(result, image, processor, model, device):
    """
    Generate mask using SAM model.

    Parameters:
    - result: YOLO prediction result.
    - image (np.array): The input image in RGB format.
    - processor (SamProcessor): SAM processor.
    - model (SamModel): SAM model.
    - device (str): 'cpu' or 'cuda'.

    Returns:
    - img_mask (np.array): The generated mask.
    """
    # Get bounding boxes and class labels from YOLO
    bboxes_labels = result.boxes.cls
    bboxes_xyxy = result.boxes.xyxy
    input_boxes = [bbox.cpu().numpy().tolist()[:4] for bbox in bboxes_xyxy]

    if len(bboxes_xyxy) == 0:
        logging.warning("No bounding boxes detected by YOLO.")
        return None

    # Prepare inputs for SAM using the processor
    inputs = processor(images=image, input_boxes=[input_boxes], return_tensors="pt").to(device)

    try:
        with torch.no_grad():
            outputs = model(**inputs)

        # Resize masks to original image size
        masks = processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )[0]

        # Initialize 2D mask for image
        img_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # Process masks
        for mask_idx, bbox_label in enumerate(bboxes_labels):
            pred_label = int(bbox_label.item())

            if mask_idx >= masks.shape[0]:
                logging.warning("Mismatch between number of masks and detections.")
                continue

            mask = masks[mask_idx]  # this should be 2D

            # Ensure mask is a 2D array
            if mask.ndim == 3:
                mask = mask[0]

            # Convert tensor to numpy array and apply binary threshold
            mask = mask.cpu().numpy()
            mask_binary = (mask > 0.5).astype(np.uint8)

            # Assign class ID to mask
            mask_labeled = mask_binary * pred_label

            # Update the image mask, resolving overlaps by overwriting
            img_mask[mask_labeled > 0] = mask_labeled[mask_labeled > 0]

    except Exception as e:
        logging.error(f"Error during SAM mask creation: {e}")
        return None

    return img_mask

def overlay_mask_on_image(image, mask):
    """
    Overlay the segmentation mask on the image for visualization.

    Parameters:
    - image (np.array): Original image in RGB format.
    - mask (np.array): Segmentation mask.

    Returns:
    - overlay (np.array): Image with mask overlay.
    """
    # Apply color map to mask
    color_mask = cv2.applyColorMap((mask * 25).astype(np.uint8), cv2.COLORMAP_JET)
    color_mask = cv2.cvtColor(color_mask, cv2.COLOR_BGR2RGB)  # Convert to RGB
    # Overlay
    overlay = cv2.addWeighted(image, 0.7, color_mask, 0.3, 0)
    return overlay

def crop_and_save_by_class(segmenter, image, mask, relative_path):
    """
    Crops the image based on class-specific masks and saves them to the crops directory.
    Optionally removes the background from the cropped images.

    Parameters:
    - segmenter (Segmenter): The Segmenter instance.
    - image (np.array): The original image.
    - mask (np.array): The segmentation mask.
    - relative_path (str): The relative path of the image for maintaining directory structure.
    """
    for class_id, class_name in CLASSES.items():
        if class_id == 0:
            continue  # Skip background class for cropping

        # Create binary mask for current class
        class_mask = (mask == class_id).astype(np.uint8) * 255

        # Find contours for class mask
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            logging.info(f"No instances found for class '{class_name}' in image '{relative_path}'.")
            continue  # No mask for this class

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)

            # Apply padding?
            padding = 0  # could make this configurable
            x = max(x - padding, 0)
            y = max(y - padding, 0)
            w = min(w + 2 * padding, image.shape[1] - x)
            h = min(h + 2 * padding, image.shape[0] - y)

            # Crop the image
            cropped_image = image[y:y+h, x:x+w]
            cropped_mask = class_mask[y:y+h, x:x+w]

            # Prepare save path for cropped classes
            crop_relative_path = os.path.join(class_name, relative_path)
            crop_save_path = os.path.splitext(os.path.join(segmenter.crops_dir, crop_relative_path))[0] + f'_{class_name}.png'
            os.makedirs(os.path.dirname(crop_save_path), exist_ok=True)

            # Save cropped image
            cv2.imwrite(crop_save_path, cropped_image)
            logging.info(f"Cropped '{class_name}' saved to '{crop_save_path}'.")

            # Background removal from cropped image if specified
            if segmenter.remove_background:
                cropped_image_bg_removed = remove_background(cropped_image, cropped_mask, segmenter.background_color)

                # Prepare save path for background-removed cropped class images
                crop_bg_removed_save_path = os.path.splitext(os.path.join(segmenter.crops_bkgd_removed_dir, crop_relative_path))[0] + f'_{class_name}.png'
                os.makedirs(os.path.dirname(crop_bg_removed_save_path), exist_ok=True)

                # Save the background-removed cropped image
                cv2.imwrite(crop_bg_removed_save_path, cropped_image_bg_removed)
                logging.info(f"Cropped '{class_name}' with background removed saved to '{crop_bg_removed_save_path}'.")

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
    if bg_color is None:
        logging.warning("bg_color is None. Defaulting to 'black'.")
        bg_color = 'black'

    if bg_color.lower() == 'white':
        background = np.full(image.shape, 255, dtype=np.uint8)
    elif bg_color.lower() == 'black':
        background = np.zeros(image.shape, dtype=np.uint8)
    else:
        raise ValueError("Unsupported background color. Choose 'white' or 'black'.")

    if mask is not None:
        # Ensure mask is single-channel
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask_binary = (mask > 0).astype(np.uint8) * 255

        # Extract foreground using mask
        masked_foreground = cv2.bitwise_and(image, image, mask=mask_binary)

        # Extract background area
        background_mask = cv2.bitwise_not(mask_binary)
        masked_background = cv2.bitwise_and(background, background, mask=background_mask)

        # Combine foreground and new background
        final_image = cv2.add(masked_foreground, masked_background)
    else:
        # If no mask provided, assume entire image is foreground
        final_image = image.copy()

    return final_image
