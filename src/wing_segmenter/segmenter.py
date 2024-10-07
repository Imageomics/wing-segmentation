import os
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from wing_segmenter.model_manager import load_models
from wing_segmenter.image_processor import process_image
from wing_segmenter.path_manager import setup_paths
from wing_segmenter.metadata_manager import generate_uuid, get_dataset_hash, get_run_hardware_info
from wing_segmenter import __version__ as package_version
from wing_segmenter.exceptions import ImageProcessingError

logging.basicConfig(level=logging.INFO)

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
        self.visualize_segmentation = config.visualize_segmentation
        self.force = config.force
        self.crop_by_class = config.crop_by_class
        self.remove_background = config.remove_background
        self.remove_bg_full = config.remove_bg_full
        if self.remove_background or self.remove_bg_full:
            self.background_color = config.background_color if config.background_color else 'black'
        else:
            self.background_color = None
        self.segmentation_info = []
        self.output_base_dir = os.path.abspath(config.outputs_base_dir) if config.outputs_base_dir else None
        self.custom_output_dir = os.path.abspath(config.custom_output_dir) if config.custom_output_dir else None

        # Generate UUID based on parameters
        self.run_uuid = generate_uuid({
            'dataset_hash': get_dataset_hash(self.dataset_path),
            'sam_model_name': self.config.sam_model,
            'yolo_model_name': self.config.yolo_model,
            'resize_mode': self.resize_mode,
            'size': self.size,
        })

        setup_paths(self)
        self.yolo_model, self.sam_model, self.sam_processor = load_models(self.config, self.device)

    def process_dataset(self):
        start_time = time.time()
        errors_occurred = False

        image_paths = []
        valid_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')
        for root, _, files in os.walk(self.dataset_path):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    image_paths.append(os.path.join(root, file))

        if not image_paths:
            logging.error("No images found in the dataset.")
            return True

        # Check for existing run unless force is specified
        if os.path.exists(self.metadata_path) and not self.force:
            with open(self.metadata_path, 'r') as f:
                existing_metadata = json.load(f)
            if existing_metadata.get('run_status', {}).get('completed'):
                logging.info(f"Processing already completed for dataset '{self.dataset_path}' with the specified parameters.")
                return False

        # Initialize metadata
        self.metadata = {
            'dataset': {
                'dataset_hash': get_dataset_hash(self.dataset_path),
                'num_images': len(image_paths)
            },
            'run_parameters': {
                'sam_model_name': self.config.sam_model,
                'yolo_model_name': self.config.yolo_model,
                'resize_mode': self.resize_mode,
                'size': self.size,
                'padding_color': self.padding_color if self.resize_mode == 'pad' else None,
                'interpolation': self.interpolation if self.size else None,
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
                            pbar.update(1)

        except Exception as e:
            logging.error(f"Processing failed: {e}")
            self.metadata['run_status']['completed'] = False
            self.metadata['run_status']['errors'] = str(e)
            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=4)
            raise e

        processing_time = time.time() - start_time
        self.metadata['run_status']['completed'] = not errors_occurred
        self.metadata['run_status']['processing_time_seconds'] = processing_time

        # Save segmentation info and CSV
        if self.segmentation_info:
            from wing_segmenter.metadata_manager import save_segmentation_info
            save_segmentation_info(self.segmentation_info, self.mask_csv)

        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=4)

        if errors_occurred:
            logging.warning(f"Processing completed with errors. Outputs are available at: \n\t {self.output_dir}")
        else:
            logging.info(f"Processing completed successfully. Outputs are available at: \n\t {self.output_dir}")

        return errors_occurred
