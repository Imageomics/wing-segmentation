import os
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor
import json
from datetime import datetime

# Mapp user input to corresponding OpenCV interpolation method
INTERPOLATION_METHODS = {
    'nearest': cv2.INTER_NEAREST,
    'linear': cv2.INTER_LINEAR,
    'cubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos4': cv2.INTER_LANCZOS4,
    'linear_exact': cv2.INTER_LINEAR_EXACT,
    'nearest_exact': cv2.INTER_NEAREST_EXACT,
}

def resize_image_task(task):
    """Helper function to resize a single image."""
    input_path, output_path, resize_dim, interpolation = task
    try:
        image = Image.open(input_path)
        image = np.array(image)
        image = cv2.resize(image, tuple(resize_dim), interpolation=interpolation)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image = Image.fromarray(image)
        image.save(output_path)
    except Exception as e:
        logging.error(f"Error processing {input_path}: {e}")

def save_metadata(output_dir, source_dir, resize_dim, interpolation, num_workers):
    """Save metadata about the resize process to a JSON file into the output directory."""
    metadata = {
        'source_directory': source_dir,
        'output_directory': output_dir,
        'resize_dimensions': resize_dim,
        'interpolation_method': interpolation,
        'num_workers': num_workers,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    logging.info(f"Metadata saved to {metadata_path}")

def resize_images(source, output, resize_dim, num_workers=1, interpolation='area'):
    """
    Resize images from the source directory, and save them to the output directory, preserving any subdirectory structure.

    Parameters:
    - source (str): Path to the source images.
    - output (str): Path to save the resized images.
    - resize_dim (list of int): [WIDTH, HEIGHT] dimensions to resize images.
    - num_workers (int): Number of parallel workers to use for resizing. Default is 1 (no parallelism).
    - interpolation (str): Interpolation method to use for resizing. Default is 'area'.
    """
    logging.info(f"Resizing images from {source} to {output} with dimensions {resize_dim} using {num_workers} workers and interpolation '{interpolation}'")

    # Get the OpenCV interpolation constant
    interpolation_method = INTERPOLATION_METHODS.get(interpolation)

    # Collect tasks for parallel execution
    tasks = []
    for root, dirs, files in os.walk(source):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')):
                # Construct the relative path and output path
                relative_path = os.path.relpath(root, source)
                output_dir = os.path.join(output, relative_path)
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_dir, os.path.splitext(file)[0] + '.png')
                tasks.append((input_path, output_path, resize_dim, interpolation_method))

    # Perform resizing in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(executor.map(resize_image_task, tasks), total=len(tasks), desc='Resizing images'))

    save_metadata(output, source, resize_dim, interpolation, num_workers)
