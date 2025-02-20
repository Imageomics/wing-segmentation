import os

def setup_paths(segmenter):
    """
    Sets up output directories and paths based on the Segmenter's configuration.

    Parameters:
    - segmenter: The Segmenter instance.
    """
    # Determine output directory
    if segmenter.custom_output_dir:
        segmenter.output_dir = os.path.abspath(segmenter.custom_output_dir)
    elif segmenter.output_base_dir:
        dataset_name = os.path.basename(segmenter.dataset_path.rstrip('/\\'))
        output_dir_name = f"{dataset_name}_{segmenter.run_uuid}"
        segmenter.output_dir = os.path.join(segmenter.output_base_dir, output_dir_name)
    else:
        dataset_name = os.path.basename(segmenter.dataset_path.rstrip('/\\'))
        output_dir_name = f"{dataset_name}_{segmenter.run_uuid}"
        segmenter.output_dir = os.path.join(os.path.dirname(segmenter.dataset_path), output_dir_name)

    os.makedirs(segmenter.output_dir, exist_ok=True)

    # Metadata directory
    segmenter.metadata_dir = os.path.join(segmenter.output_dir, 'metadata')
    os.makedirs(segmenter.metadata_dir, exist_ok=True)

    # Metadata file paths
    segmenter.metadata_json_path = os.path.join(segmenter.metadata_dir, 'run_metadata.json')
    segmenter.detection_csv_path = os.path.join(segmenter.metadata_dir, 'detection.csv')
    segmenter.coco_annotations_path = os.path.join(segmenter.metadata_dir, 'coco_annotations.json')

    # Define subdirectories that are always present
    segmenter.masks_dir = os.path.join(segmenter.output_dir, 'masks')
    segmenter.logs_dir = os.path.join(segmenter.output_dir, 'logs')
    os.makedirs(segmenter.masks_dir, exist_ok=True)
    os.makedirs(segmenter.logs_dir, exist_ok=True)

    # Conditionally create directories based on flags/options
    if segmenter.size:
        segmenter.resized_dir = os.path.join(segmenter.output_dir, 'resized')
        os.makedirs(segmenter.resized_dir, exist_ok=True)

    if segmenter.visualize_segmentation:
        segmenter.viz_dir = os.path.join(segmenter.output_dir, 'seg_viz')
        os.makedirs(segmenter.viz_dir, exist_ok=True)

    if segmenter.crop_by_class:
        segmenter.crops_dir = os.path.join(segmenter.output_dir, 'crops')
        os.makedirs(segmenter.crops_dir, exist_ok=True)

    if segmenter.remove_crops_background:
        segmenter.crops_bkgd_removed_dir = os.path.join(segmenter.output_dir, 'crops_bkgd_removed')
        os.makedirs(segmenter.crops_bkgd_removed_dir, exist_ok=True)

    if segmenter.remove_full_background:
        segmenter.full_bkgd_removed_dir = os.path.join(segmenter.output_dir, 'full_bkgd_removed')
        os.makedirs(segmenter.full_bkgd_removed_dir, exist_ok=True)

    segmenter.detection_csv = segmenter.detection_csv_path
