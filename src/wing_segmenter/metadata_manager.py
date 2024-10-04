import uuid
import hashlib
import json
import os
from wing_segmenter.constants import CLASSES

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

def get_run_hardware_info(device, num_workers):
    """
    Retrieves information about the hardware used for the run.
    
    Parameters:
    - device (str): 'cpu' or 'cuda'.
    - num_workers (int): Number of worker threads.
    
    Returns:
    - dict: Hardware information.
    """
    hardware_info = {
        'device': device,
        'num_workers': num_workers
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

def save_segmentation_info(segmentation_info, mask_csv):
    """
    Saves the segmentation information to a CSV file.
    
    Parameters:
    - segmentation_info (list): The segmentation information.
    - mask_csv (str): Path to the CSV file.
    """
    import pandas as pd
    df = pd.DataFrame(segmentation_info)
    df.to_csv(mask_csv, index=False)
