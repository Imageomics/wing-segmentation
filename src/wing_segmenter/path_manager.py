import os
from wing_segmenter.constants import CLASSES 

def setup_paths(segmenter):
    """
    Sets up output directories and paths based on the Segmenter's configuration.

    Parameters:
    - segmenter: The Segmenter instance.
    """
    # Determine output directory
    if segmenter.custom_output_dir:
        # User has specified a fully custom output directory
        segmenter.output_dir = os.path.abspath(segmenter.custom_output_dir)
    elif segmenter.output_base_dir:
        # Create output directory based on base directory and run UUID
        dataset_name = os.path.basename(segmenter.dataset_path.rstrip('/\\'))
        output_dir_name = f"{dataset_name}_{segmenter.run_uuid}"
        segmenter.output_dir = os.path.join(segmenter.output_base_dir, output_dir_name)
    else:
        # Default: Create output directory in the parent of the dataset path
        dataset_name = os.path.basename(segmenter.dataset_path.rstrip('/\\'))
        output_dir_name = f"{dataset_name}_{segmenter.run_uuid}"
        segmenter.output_dir = os.path.join(os.path.dirname(segmenter.dataset_path), output_dir_name)

    # Create the root output directory
    os.makedirs(segmenter.output_dir, exist_ok=True)

    # Metadata file path
    segmenter.metadata_path = os.path.join(segmenter.output_dir, 'metadata.json')

    # Define subdirectories
    segmenter.resized_dir = os.path.join(segmenter.output_dir, 'resized')
    segmenter.masks_dir = os.path.join(segmenter.output_dir, 'masks')
    segmenter.viz_dir = os.path.join(segmenter.output_dir, 'seg_viz')
    segmenter.crops_dir = os.path.join(segmenter.output_dir, 'crops')
    segmenter.logs_dir = os.path.join(segmenter.output_dir, 'logs')

    # Create subdirectories
    os.makedirs(segmenter.resized_dir, exist_ok=True)
    os.makedirs(segmenter.masks_dir, exist_ok=True)
    os.makedirs(segmenter.viz_dir, exist_ok=True)
    os.makedirs(segmenter.crops_dir, exist_ok=True)
    os.makedirs(segmenter.logs_dir, exist_ok=True)

    # Create directories for background removal if needed
    if segmenter.remove_background and segmenter.save_intermediates:
        segmenter.crops_bkgd_removed_dir = os.path.join(segmenter.output_dir, 'crops_bkgd_removed')
        segmenter.full_bkgd_removed_dir = os.path.join(segmenter.output_dir, 'full_bkgd_removed')
        os.makedirs(segmenter.crops_bkgd_removed_dir, exist_ok=True)
        os.makedirs(segmenter.full_bkgd_removed_dir, exist_ok=True)

    # Additional directories based on resizing options
    if segmenter.save_intermediates and segmenter.size:
        os.makedirs(segmenter.resized_dir, exist_ok=True)

    # CSV file for segmentation info
    segmenter.mask_csv = os.path.join(segmenter.output_dir, 'segmentation.csv')
