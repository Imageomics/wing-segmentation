import os
import uuid
from wing_segmenter.constants import CLASSES 

def setup_paths(segmenter):
    """
    Sets up output directories and paths based on the Segmenter's configuration.

    Parameters:
    - segmenter: The Segmenter instance.
    """
    # Create output directory
    dataset_name = os.path.basename(segmenter.dataset_path.rstrip('/\\'))
    output_dir_name = f"{dataset_name}_{segmenter.run_uuid}"
    if segmenter.output_base_dir:
        segmenter.output_dir = os.path.join(segmenter.output_base_dir, output_dir_name)
    else:
        segmenter.output_dir = os.path.join(os.path.dirname(segmenter.dataset_path), output_dir_name)
    os.makedirs(segmenter.output_dir, exist_ok=True)

    # Metadata file path
    segmenter.metadata_path = os.path.join(segmenter.output_dir, 'metadata.json')

    # Prepare output subdirectories
    segmenter.resized_dir = os.path.join(segmenter.output_dir, 'resized')
    segmenter.masks_dir = os.path.join(segmenter.output_dir, 'masks')
    segmenter.viz_dir = os.path.join(segmenter.output_dir, 'seg_viz')
    segmenter.crops_dir = os.path.join(segmenter.output_dir, 'crops')
    segmenter.logs_dir = os.path.join(segmenter.output_dir, 'logs')
    os.makedirs(segmenter.masks_dir, exist_ok=True)
    os.makedirs(segmenter.viz_dir, exist_ok=True)
    os.makedirs(segmenter.crops_dir, exist_ok=True)
    os.makedirs(segmenter.logs_dir, exist_ok=True)


    # Create directory for images with background removed if required
    if segmenter.remove_background and segmenter.save_intermediates:
        # Create directory for background-removed cropped images
        segmenter.crops_bkgd_removed_dir = os.path.join(segmenter.output_dir, 'crops_bkgd_removed')
        os.makedirs(segmenter.crops_bkgd_removed_dir, exist_ok=True)

        # Create directory for background-removed full images
        segmenter.full_bkgd_removed_dir = os.path.join(segmenter.output_dir, 'full_bkgd_removed')
        os.makedirs(segmenter.full_bkgd_removed_dir, exist_ok=True)

    # If resizing is enabled, create resized directories
    if segmenter.save_intermediates and segmenter.size:
        os.makedirs(segmenter.resized_dir, exist_ok=True)

    # Mask CSV path
    segmenter.mask_csv = os.path.join(segmenter.output_dir, 'segmentation.csv')
