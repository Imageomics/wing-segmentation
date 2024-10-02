import os
import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from wing_segmenter.model_manager import load_models
from wing_segmenter.resizer import resize_image
from wing_segmenter.path_manager import setup_paths
from wing_segmenter.image_processor import process_image
from wing_segmenter.metadata_manager import (
    generate_uuid,
    get_dataset_hash,
    get_run_hardware_info,
)
from wing_segmenter.exceptions import ModelLoadError, ImageProcessingError
from wing_segmenter import __version__ as package_version

class Segmenter:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.dataset_path = os.path.abspath(config.dataset)
        self.size = config.size
        self.resize_mode = config.resize_mode
        self.padding_color = config.padding_color
        self.interpolation = config.interpolation
        self.num_workers = config.num_workers
        self.save_intermediates = config.save_intermediates
        self.visualize_segmentation = config.visualize_segmentation
        self.force = config.force
        self.crop_by_class = config.crop_by_class
        self.remove_background = config.remove_background
        self.remove_bg_full = config.remove_bg_full
        self.background_color = config.background_color if (self.remove_background or self.remove_bg_full) else None
        self.segmentation_info = []
        self.output_base_dir = os.path.abspath(config.output_dir) if config.output_dir else None

        # Define your namespace UUID based on a string
        self.NAMESPACE_UUID = uuid.uuid5(uuid.NAMESPACE_DNS, 'Imageomics Wing Segmentation')

        # Handle resizing dimensions
        if self.size:
            if len(self.size) == 1:
                self.width = self.height = self.size[0]
            elif len(self.size) == 2:
                self.width, self.height = self.size
            else:
                raise ValueError("Invalid size argument. Size must have either one or two values.")
        else:
            self.width = self.height = None  # if no resizing use None

        # Prepare parameters for hashing
        self.parameters_for_hash = {
            'dataset_hash': get_dataset_hash(self.dataset_path),
            'sam_model_name': self.config.sam_model,
            'yolo_model_name': self.config.yolo_model,
            'resize_mode': self.resize_mode,
            'size': self.size if self.size else None,
            'width': self.width,
            'height': self.height,
            'padding_color': self.padding_color if self.resize_mode == 'pad' else None,
            'interpolation': self.interpolation if self.size else None,
            'save_intermediates': self.save_intermediates,
            'visualize_segmentation': self.visualize_segmentation,
            'crop_by_class': self.crop_by_class,
            'remove_background': self.remove_background,
            'remove_bg_full': self.remove_bg_full,
            'background_color': self.background_color
        }

        # Generate UUID based on parameters
        self.run_uuid = generate_uuid(self.parameters_for_hash, self.NAMESPACE_UUID)

        setup_paths(self)

        self.yolo_model, self.sam_model, self.sam_processor = load_models(self.config, self.device)

    def process_dataset(self):
        start_time = time.time() 
        errors_occurred = False

        # Prepare image paths
        image_paths = []
        valid_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')
        for root, _, files in os.walk(self.dataset_path):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    image_paths.append(os.path.join(root, file))

        if not image_paths:
            logging.error("No images found in the dataset.")
            errors_occurred = True
            return errors_occurred

        # Check for existing run unless force is specified
        if os.path.exists(self.metadata_path) and not self.force:
            with open(self.metadata_path, 'r') as f:
                existing_metadata = json.load(f)
            if existing_metadata.get('run_status', {}).get('completed'):
                logging.info(f"Processing already completed for dataset '{self.dataset_path}' with the specified parameters.")
                logging.info(f"Outputs are available at '{self.output_dir}'.")
                return errors_occurred

        # Initialize metadata
        self.metadata = {
            'dataset': {
                'dataset_hash': self.parameters_for_hash['dataset_hash'],
                'num_images': len(image_paths)
            },
            'run_parameters': {
                'sam_model_name': self.config.sam_model,
                'yolo_model_name': self.config.yolo_model,
                'resize_mode': self.resize_mode,
                'size': self.size if self.size else None,
                'padding_color': self.padding_color if self.resize_mode == 'pad' else None,
                'interpolation': self.interpolation if self.size else None,
                'save_intermediates': self.save_intermediates,
                'visualize_segmentation': self.visualize_segmentation,
                'crop_by_class': self.crop_by_class,
                'remove_background': self.remove_background,
                'remove_bg_full': self.remove_bg_full, 
                'background_color': self.background_color
            },
            'run_hardware': get_run_hardware_info(self.device, self.num_workers),
            'run_status': {
                'completed': False,
                'processing_time_seconds': None,
                'package_version': package_version,
                'errors': None
            }
        }
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=4)

        try:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {executor.submit(process_image, self, image_path): image_path for image_path in image_paths}

                with tqdm(total=len(futures), desc='Processing Images', unit='image') as pbar:
                    for future in as_completed(futures):
                        image_path = futures[future]
                        try:
                            future.result()
                        except ImageProcessingError:
                            errors_occurred = True
                            self.metadata['run_status']['errors'] = "One or more images failed during processing."
                        finally:
                            pbar.update(1)  # update progress for each completed task

        except Exception as e:
            logging.error(f"Processing failed: {e}")
            self.metadata['run_status']['completed'] = False
            self.metadata['run_status']['errors'] = str(e)
            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=4)
            raise e

        # Save segmentation info
        if self.segmentation_info:
            from wing_segmenter.metadata_manager import save_segmentation_info
            save_segmentation_info(self.segmentation_info, self.mask_csv)

        # Update metadata on completion
        processing_time = time.time() - start_time

        self.metadata['run_status']['completed'] = not errors_occurred
        self.metadata['run_status']['processing_time_seconds'] = processing_time

        # Save updated metadata
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=4)

        if errors_occurred:
            logging.warning(f"Processing completed with errors. Outputs are available at: \n\t{self.output_dir}")
        else:
            logging.info(f"Processing completed successfully. Outputs are available at: \n\t{self.output_dir}")

        return errors_occurred
