import logging
import torch
import subprocess
import re
import os
from ultralytics import YOLO
from transformers import SamModel, SamProcessor
from huggingface_hub import hf_hub_download
from wing_segmenter.exceptions import ModelLoadError

logging.getLogger("ultralytics").setLevel(logging.WARNING)

def get_gpu_with_lowest_memory(min_required_memory_mb=1024):
    """Returns the GPU ID with the lowest memory usage that has at least `min_required_memory_mb` free memory."""
    try:
        # Run nvidia-smi to get the memory usage of each GPU
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free,memory.total', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        if result.returncode != 0:
            logging.error("nvidia-smi failed to run")
            return None

        # Parse the output
        memory_info = result.stdout.strip().split('\n')
        gpu_memory = []
        for idx, info in enumerate(memory_info):
            free, total = map(int, re.findall(r'\d+', info))
            if free >= min_required_memory_mb:  # check if the GPU has enough free memory
                utilization = free / total
                gpu_memory.append((idx, free, utilization))  # (GPU index, free memory, utilization percentage)

        # If no GPUs with enough memory are found
        if not gpu_memory:
            logging.error(f"No GPUs with at least {min_required_memory_mb} MB of free memory found.")
            return None

        # Sort GPUs by memory usage (most free memory first)
        gpu_memory.sort(key=lambda x: (-x[1], x[2]))  # sort by free memory, then use percentage
        selected_gpu = gpu_memory[0][0]  # select GPU with most free memory

        return selected_gpu

    except Exception as e:
        logging.error(f"Error selecting GPU with sufficient memory: {e}")
        return None

def get_yolo_model(model_name, device):
    logging.info(f"Loading YOLO model: {model_name}")
    try:
        # Determine device to use
        if device == 'cuda':
            gpu_id = get_gpu_with_lowest_memory(min_required_memory_mb=10000)
            if gpu_id is not None:
                logging.info(f"Using GPU {gpu_id} with the most free memory.")
                torch.cuda.set_device(gpu_id)
                device_spec = f"cuda:{gpu_id}"
            else:
                logging.warning("Could not find a suitable GPU. Using the default GPU.")
                device_spec = "cuda"
        else:
            device_spec = 'cpu'

        # Load the YOLO model from Hugging Face or a local path
        if ':' in model_name:
            repo_id, filename = model_name.split(':')
            model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        else:
            model_path = model_name

        # Use the YOLO class to load the model from the path
        model = YOLO(model_path)
        
        # Explicitly move model to the specified device
        model.to(device_spec)

        logging.info(f"YOLO model loaded onto {device_spec}")

    except Exception as e:
        logging.error(f"Failed to load YOLO model '{model_name}': {e}")
        raise ModelLoadError(f"Failed to load YOLO model '{model_name}': {e}")

    return model

def get_sam_model_and_processor(sam_model_name, device):
    """
    Load SAM model and processor from Hugging Face.

    Parameters:
    - sam_model_name (str): Hugging Face model identifier for SAM.
    - device (str): 'cpu' or 'cuda'.

    Returns:
    - sam_model (SamModel): Loaded SAM model.
    - sam_processor (SamProcessor): Loaded SAM processor.
    """
    logging.info(f"Loading SAM model: {sam_model_name}")
    try:
        # For SAM model use device_map
        if device.startswith('cuda'):
            device_map = device
        else:
            device_map = 'cpu'
        
        # Load model with explicit device map
        sam_model = SamModel.from_pretrained(
            sam_model_name, 
            device_map=device_map
        )
        
        # Load processor
        sam_processor = SamProcessor.from_pretrained(sam_model_name)
        
        logging.info(f"Loaded SAM model and processor successfully on {device}.")
    except Exception as e:
        logging.error(f"Failed to load SAM model '{sam_model_name}': {e}")
        raise ModelLoadError(f"Failed to load SAM model '{sam_model_name}': {e}")
    
    return sam_model, sam_processor

def load_models(config, device):
    """
    Loads both YOLO and SAM models based on the configuration.

    Parameters:
    - config: The configuration object containing model names.
    - device (str): The device to load models onto ('cpu' or 'cuda').

    Returns:
    - yolo_model: The loaded YOLO model.
    - sam_model: The loaded SAM model.
    - sam_processor: The loaded SAM processor.
    """
    try:
        yolo_model = get_yolo_model(config.yolo_model, device)
        sam_model, sam_processor = get_sam_model_and_processor(config.sam_model, device)
    except ModelLoadError as e:
        logging.error(f"Failed to load models: {e}")
        raise
    return yolo_model, sam_model, sam_processor
