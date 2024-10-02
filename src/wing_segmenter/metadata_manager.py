import uuid
import hashlib
import json
import os

def generate_uuid(parameters, namespace_uuid):
    """
    Generates a UUID based on the hash of the parameters.
    
    Parameters:
    - parameters (dict): The parameters to hash.
    - namespace_uuid (uuid.UUID): The namespace UUID.
    
    Returns:
    - uuid.UUID: The generated UUID.
    """
    # Convert parameters to a sorted JSON string to ensure consistency
    param_str = json.dumps(parameters, sort_keys=True).encode('utf-8')
    return uuid.uuid5(namespace_uuid, hashlib.sha256(param_str).hexdigest())

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

def update_segmentation_info(segmentation_info, image_path, mask):
    """
    Updates the segmentation information list with details about the processed image.
    
    Parameters:
    - segmentation_info (list): The list to update.
    - image_path (str): Path to the processed image.
    - mask (np.array): The segmentation mask.
    """
    segmentation_info.append({
        'image_path': image_path,
        'mask': mask.tolist()
    })

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
