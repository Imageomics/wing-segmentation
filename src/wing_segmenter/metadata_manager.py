import uuid
import hashlib
import json
import os
from wing_segmenter.constants import CLASSES
import pycocotools.mask as mask_util
import numpy as np
import logging

NAMESPACE_UUID = uuid.UUID('00000000-0000-0000-0000-000000000000')

def generate_uuid(parameters):
    """
    Generates a UUID based on the provided parameters and a fixed namespace UUID.
    
    Parameters:
    - parameters (dict): The parameters to hash.
    
    Returns:
    - uuid.UUID: The generated UUID.
    """
    # Serialize parameters to a sorted JSON string to ensure consistency
    param_str = json.dumps(parameters, sort_keys=True)
    return uuid.uuid5(NAMESPACE_UUID, param_str)

def get_dataset_hash(dataset_path):
    """
    Generates a hash for the dataset by hashing all file paths and their sizes.
    
    Parameters:
    - dataset_path (str): Path to the dataset.
    
    Returns:
    - str: The dataset hash.
    """
    hash_md5 = hashlib.md5()
    for root, dirs, files in os.walk(dataset_path):
        for file in sorted(files):
            file_path = os.path.join(root, file)
            try:
                hash_md5.update(file_path.encode('utf-8'))
                hash_md5.update(str(os.path.getsize(file_path)).encode('utf-8'))
            except FileNotFoundError:
                continue
    return hash_md5.hexdigest()

def get_run_hardware_info(device):
    """
    Retrieves information about the hardware used for the run.
    
    Parameters:
    - device (str): 'cpu' or 'cuda'.
    
    Returns:
    - dict: Hardware information.
    """
    hardware_info = {
        'device': device,
    }
    if device == 'cuda':
        import torch
        hardware_info['cuda_device'] = torch.cuda.get_device_name(0)
        hardware_info['cuda_memory_total'] = torch.cuda.get_device_properties(0).total_memory
    return hardware_info

def update_segmentation_info(segmentation_info, image_path, classes_present):
    """
    Updates the segmentation information list with binary flags for each class.
    
    Parameters:
    - segmentation_info (list): The list to update.
    - image_path (str): Path to the processed image.
    - classes_present (list): List of class names detected in the image.
    """
    entry = {'image': image_path}
    
    for class_id, class_name in CLASSES.items():
        # Assign 1 if the class is present, else 0
        entry[class_name] = 1 if class_name in classes_present else 0
    
    segmentation_info.append(entry)

def save_segmentation_info(segmentation_info, detection_csv_path):
    """
    Saves the segmentation information to a CSV file.

    Parameters:
    - segmentation_info (list): The segmentation information.
    - detection_csv_path (str): Path to the CSV file.
    """
    import pandas as pd
    if not segmentation_info:
        logging.warning(f"No segmentation information to save at '{detection_csv_path}'.")
        return
    df = pd.DataFrame(segmentation_info)
    df.to_csv(detection_csv_path, index=False)
    logging.info(f"Saved detection information to '{detection_csv_path}'.")

def add_coco_image_info(coco_annotations_path, relative_path, image_shape):
    """
    Adds image information to the COCO annotations file.

    Parameters:
    - coco_annotations_path (str): Path to the coco_annotations.json file.
    - relative_path (str): Relative path to the image.
    - image_shape (tuple): Shape of the image (height, width, channels).

    Returns:
    - int: The image_id assigned to this image.
    """
    height, width = image_shape[:2]
    image_id = None

    # Load existing annotations or initialize
    if os.path.exists(coco_annotations_path):
        with open(coco_annotations_path, 'r') as f:
            coco = json.load(f)
    else:
        coco = {
            "images": [],
            "annotations": [],
            "categories": []
        }
        # Initialize categories
        for class_id, class_name in CLASSES.items():
            if class_id == 0:
                continue  # Skip background
            coco['categories'].append({
                "id": class_id,
                "name": class_name,
                "supercategory": "none"
            })

    # Check if image already exists
    for img in coco['images']:
        if img['file_name'] == relative_path:
            image_id = img['id']
            break

    if image_id is None:
        image_id = len(coco['images']) + 1
        coco['images'].append({
            "id": image_id,
            "file_name": relative_path,
            "height": height,
            "width": width
        })

    # Save back
    with open(coco_annotations_path, 'w') as f:
        json.dump(coco, f, indent=4)

    return image_id

def add_coco_annotation(coco_annotations_path, image_id, category_id, bbox, mask):
    """
    Adds an annotation to the COCO annotations file.

    Parameters:
    - coco_annotations_path (str): Path to the coco_annotations.json file.
    - image_id (int): ID of the image.
    - category_id (int): ID of the category.
    - bbox (list): Bounding box [x1, y1, x2, y2].
    - mask (np.array): Binary mask for the annotation.
    """
    if not os.path.exists(coco_annotations_path):
        raise FileNotFoundError(f"COCO annotations file '{coco_annotations_path}' does not exist.")

    with open(coco_annotations_path, 'r') as f:
        coco = json.load(f)

    annotation_id = len(coco['annotations']) + 1

    # Convert bbox from [x1, y1, x2, y2] to [x, y, width, height]
    x, y, x2, y2 = bbox
    width = x2 - x
    height = y2 - y
    bbox_coco = [x, y, width, height]

    # Encode mask to RLE; TODO: make polygon optional
    rle = mask_util.encode(np.asfortranarray(mask))
    rle['counts'] = rle['counts'].decode('utf-8')  # Convert bytes to string

    annotation = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox_coco,
        "area": width * height,
        "segmentation": rle,
        "iscrowd": 0
    }

    coco['annotations'].append(annotation)

    with open(coco_annotations_path, 'w') as f:
        json.dump(coco, f, indent=4)
    
    logging.debug(f"Added COCO annotation ID {annotation_id} for image ID {image_id}.")
