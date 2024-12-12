import argparse

def main():
    parser = argparse.ArgumentParser(
        prog='wingseg',
        description="Wing Segmenter CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    subparsers = parser.add_subparsers(title='Commands', dest='command', required=True)

    # Subcommand: segment
    segment_parser = subparsers.add_parser(
        'segment',
        help='Segment images and store segmentation masks.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required argument
    segment_parser.add_argument('--dataset', required=True, help='Path to dataset images')

    # Resizing options
    resize_group = segment_parser.add_argument_group('Resizing Options')

    # Dimension specifications
    resize_group.add_argument('--size', nargs='+', type=int,
                              help='Target size. Provide one value for square dimensions or two for width and height.')

    # Resizing mode
    resize_group.add_argument('--resize-mode', choices=['distort', 'pad'], default=None,
                              help='Resizing mode. "distort" resizes without preserving aspect ratio, "pad" preserves aspect ratio and adds padding if necessary. Required with --size.')

    # Padding options (to preserve aspect ratio)
    resize_group.add_argument('--padding-color', choices=['black', 'white'], default=None,
                              help='Padding color to use when --resize-mode is "pad".')

    # Interpolation options
    resize_group.add_argument('--interpolation', choices=['nearest', 'linear', 'cubic', 'area', 'lanczos4', 'linear_exact', 'nearest_exact'],
                              default='area',
                              help='Interpolation method to use when resizing. For upscaling, "lanczos4" is recommended.')

    # Bounding box padding option
    bbox_group = segment_parser.add_argument_group('Bounding Box Options')
    bbox_group.add_argument('--bbox-padding', type=int, default=None,
                            help='Padding to add to bounding boxes in pixels. Defaults to no padding.')


    # Output options within mutually exclusive group
    output_group = segment_parser.add_mutually_exclusive_group()
    output_group.add_argument('--outputs-base-dir', default=None, help='Base path to store outputs.')
    output_group.add_argument('--custom-output-dir', default=None, help='Fully custom directory to store all output files.')

    # General processing options
    segment_parser.add_argument('--sam-model', default='facebook/sam-vit-base',
                                help='SAM model to use (e.g., facebook/sam-vit-base)')
    segment_parser.add_argument('--yolo-model', default='imageomics/butterfly_segmentation_yolo_v8:yolov8m_shear_10.0_scale_0.5_translate_0.1_fliplr_0.0_best.pt',
                                help='YOLO model to use (local path or Hugging Face repo).')
    segment_parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu',
                                help='Device to use for processing.')
    segment_parser.add_argument('--visualize-segmentation', action='store_true',
                                help='Generate and save segmentation visualizations.')
    segment_parser.add_argument('--crop-by-class', action='store_true',
                                help='Enable cropping of segmented classes into crops/ directory.')
    segment_parser.add_argument('--force', action='store_true',
                                help='Force reprocessing even if outputs already exist.')

    # Background removal options
    bg_group = segment_parser.add_argument_group('Background Removal Options')
    bg_group.add_argument('--remove-crops-background', action='store_true',
                           help='Remove background from cropped images.')
    bg_group.add_argument('--remove-full-background', action='store_true',
                           help='Remove background from the entire (resized or original) image.')
    bg_group.add_argument('--background-color', choices=['white', 'black'], default=None,
                           help='Background color to use when removing background.')

    # Subcommand: scan-runs
    scan_parser = subparsers.add_parser('scan-runs', help='List existing processing runs for a dataset.')
    scan_parser.add_argument('--dataset', required=True, help='Path to the dataset directory.')
    scan_parser.add_argument('--output-dir', default=None, help='Base path where outputs were stored.')

    # Parse arguments
    args = parser.parse_args()

    # Command input validations
    if args.command == 'segment':
        # If size is provided, enforce resizing options
        if args.size:
            if len(args.size) not in [1, 2]:
                parser.error('--size must accept either one value (square resize) or two values (width and height).')
            if not args.resize_mode:
                parser.error('--resize-mode must be specified when --size is provided.')
        # If no size is provided, ensure that resizing options were not explicitly set
        else:
            if args.resize_mode is not None:
                parser.error('Resizing options (--resize-mode) require --size to be specified.')
            if args.padding_color is not None:
                parser.error('Resizing options (--padding-color) require --size to be specified.')

        # --remove-crops-background requires --crop-by-class
        if args.remove_crops_background and not args.crop_by_class:
            parser.error('--remove-crops-background requires --crop-by-class to be set.')

        # Need to set croped or full background removal to set background color
        if args.background_color and not (args.remove_crops_background or args.remove_full_background):
            parser.error('--background-color can only be set when background removal is enabled.')

        # Ensure that if --custom-output-dir is set, --outputs-base-dir is not used
        if args.custom_output_dir and args.outputs_base_dir:
            parser.error('Cannot specify both --outputs-base-dir and --custom-output-dir. Choose one.')

        # Validate bbox-padding
        if args.bbox_padding is not None and args.bbox_padding < 0:
            parser.error('--bbox-padding must be a non-negative integer.')

    # Execute the subcommand
    if args.command == 'segment':
        from wing_segmenter.segmenter import Segmenter

        segmenter = Segmenter(args)
        segmenter.process_dataset()

    elif args.command == 'scan-runs':
        from wing_segmenter.run_scanner import scan_runs

        scan_runs(dataset_path=args.dataset, output_base_dir=args.output_dir)
