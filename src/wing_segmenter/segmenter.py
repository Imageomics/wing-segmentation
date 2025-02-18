import os
import json
import logging
from PIL import Image
import time
from tqdm import tqdm

from wing_segmenter.model_manager import load_models
from wing_segmenter.image_processor import process_image, get_class_color_map
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
        self.visualize_segmentation = config.visualize_segmentation
        self.force = config.force
        self.crop_by_class = config.crop_by_class
        self.remove_crops_background = config.remove_crops_background
        self.remove_full_background = config.remove_full_background
        if self.remove_crops_background or self.remove_full_background:
            self.background_color = config.background_color if config.background_color else 'black'
        else:
            self.background_color = None
        self.segmentation_info = []
        self.output_base_dir = os.path.abspath(config.outputs_base_dir) if config.outputs_base_dir else None
        self.custom_output_dir = os.path.abspath(config.custom_output_dir) if config.custom_output_dir else None
        self.bbox_padding = config.bbox_padding if config.bbox_padding is not None else 0
        self.class_color_map = get_class_color_map()

        # Generate UUID based on parameters
        self.run_uuid = generate_uuid({
            'dataset_hash': get_dataset_hash(self.dataset_path),
            'sam_model_name': self.config.sam_model,
            'yolo_model_name': self.config.yolo_model,
            'resize_mode': self.resize_mode,
            'size': self.size,
            'interpolation': self.interpolation if self.size else None,
            'bbox_padding': self.bbox_padding,
        })

        setup_paths(self)
        self.yolo_model, self.sam_model, self.sam_processor = load_models(self.config, self.device)

    @staticmethod
    def is_valid_image(file_path):
        """
        Returns True if the file at file_path can be opened and verified as an image.
        """
        try:
            with Image.open(file_path) as img:
                img.verify()
            return True
        except Exception:
            return False

    def process_dataset(self):
        start_time = time.time()
        errors_occurred = False

        image_paths = []
        for root, _, files in os.walk(self.dataset_path):
            for file in files:
                full_path = os.path.join(root, file)
                if self.is_valid_image(full_path):
                    image_paths.append(full_path)

        if not image_paths:
            logging.error("No images found in the dataset.")
            return True

        # Check for existing run unless force is specified
        if os.path.exists(self.metadata_json_path) and not self.force:
            with open(self.metadata_json_path, 'r') as f:
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
                'bbox_pad_px': self.bbox_padding,
                'resize_padding_color': self.padding_color if self.resize_mode == 'pad' else None,
                'interpolation': self.interpolation if self.size else None,
                'visualize_segmentation': self.visualize_segmentation,
                'crop_by_class': self.crop_by_class,
                'remove_crops_background': self.remove_crops_background,
                'remove_full_background': self.remove_full_background,
                'background_color': self.background_color
            },
            'run_hardware': get_run_hardware_info(self.device),
            'run_status': {
                'completed': False,
                'processing_time_seconds': None,
                'package_version': package_version,
                'errors': None
            }
        }
        with open(self.metadata_json_path, 'w') as f:
            json.dump(self.metadata, f, indent=4)

        # Log output directory information
        logging.info(f"Processing {len(image_paths)} images")
        logging.info(f"Output directory: {self.output_dir}")

        try:
            for image_path in tqdm(image_paths, desc='Processing Images', unit='image'):
                try:
                    process_image(self, image_path)
                except ImageProcessingError:
                    errors_occurred = True
                    self.metadata['run_status']['errors'] = "One or more images failed during processing."

        except Exception as e:
            logging.error(f"Processing failed: {e}")
            self.metadata['run_status']['completed'] = False
            self.metadata['run_status']['errors'] = str(e)
            with open(self.metadata_json_path, 'w') as f:
                json.dump(self.metadata, f, indent=4)
            raise e

        processing_time = time.time() - start_time
        self.metadata['run_status']['completed'] = not errors_occurred
        self.metadata['run_status']['processing_time_seconds'] = processing_time

        # Save detection info and CSV
        if self.segmentation_info:
            from wing_segmenter.metadata_manager import save_segmentation_info
            save_segmentation_info(self.segmentation_info, self.detection_csv_path)

        with open(self.metadata_json_path, 'w') as f:
            json.dump(self.metadata, f, indent=4)

        if errors_occurred:
            logging.warning(f"Processing completed with errors. Outputs are available at: \n\t {self.output_dir}")
        else:
            logging.info(f"Processing completed successfully. Outputs are available at: \n\t {self.output_dir}")

        return errors_occurred
