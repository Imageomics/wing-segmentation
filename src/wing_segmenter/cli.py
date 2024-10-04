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
                              help='Resizing mode. "distort" resizes without preserving aspect ratio, "pad" preserves aspect ratio and adds padding if necessary.')

    # Padding options
    resize_group.add_argument('--padding-color', choices=['black', 'white'], default='black',
                              help='Padding color to use when --resize-mode is "pad".')

    # Interpolation options
    resize_group.add_argument('--interpolation', choices=['nearest', 'linear', 'cubic', 'area', 'lanczos4', 'linear_exact', 'nearest_exact'],
                              default='area',
                              help='Interpolation method to use when resizing. For upscaling, "lanczos4" is recommended.')

    # Output options within mutually exclusive group
    output_group = segment_parser.add_mutually_exclusive_group()
    output_group.add_argument('--outputs-base-dir', default=None, help='Base path to store outputs.')
    output_group.add_argument('--custom-output-dir', default=None, help='Fully custom directory to store all output files.')

    # General processing options
    segment_parser.add_argument('--sam-model', default='facebook/sam-vit-base',
                                help='SAM model to use (e.g., facebook/sam-vit-base)')
    segment_parser.add_argument('--yolo-model', default='imageomics/butterfly_segmentation_yolo_v8:yolov8m_shear_10.0_scale_0.5_translate_0.1_fliplr_0.0_best.pt',
                                help='YOLO model to use (local path or Hugging Face repo).')
    segment_parser.add_argument('--num-workers', type=int, default=1,
                                help='Number of worker threads to use for processing.')
    segment_parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu',
                                help='Device to use for processing.')
    segment_parser.add_argument('--save-intermediates', action='store_true',
                                help='Save intermediate files (resized images and segmentation masks).')
    segment_parser.add_argument('--visualize-segmentation', action='store_true',
                                help='Generate and save segmentation visualizations.')
    segment_parser.add_argument('--force', action='store_true',
                                help='Force reprocessing even if outputs already exist.')
    segment_parser.add_argument('--crop-by-class', action='store_true',
                                help='Enable cropping of segmented classes into crops/ directory.')

    # CLI Flags for background removal
    bg_group = segment_parser.add_argument_group('Background Removal Options')
    bg_group.add_argument('--remove-background', action='store_true', default=False,
                           help='Remove background from cropped images.')
    bg_group.add_argument('--background-color', choices=['white', 'black'], default='black',
                           help='Background color to use when removing background. Applicable only if --remove-background is set.')
    bg_group.add_argument('--remove-bg-full', action='store_true',
                           help='Remove background from the entire (resized or original) image.')

    # Subcommand: scan-runs
    scan_parser = subparsers.add_parser('scan-runs', help='List existing processing runs for a dataset.')
    scan_parser.add_argument('--dataset', required=True, help='Path to the dataset directory.')
    scan_parser.add_argument('--output-dir', default=None, help='Base path where outputs were stored.')

    # Parse arguments
    args = parser.parse_args()

    # Validation for resizing options
    if args.command == 'segment':
        # If size is provided, enforce resizing options
        if args.size:
            if len(args.size) not in [1, 2]:
                parser.error('--size must accept either one value (square resize) or two values (width and height).')
            if not args.resize_mode:
                parser.error('--resize-mode must be specified when --size is provided.')
            if args.resize_mode == 'pad' and args.padding_color is None:
                args.padding_color = 'black'
        # If no size is provided, ensure that resizing options were not explicitly set
        else:
            if args.resize_mode is not None or args.padding_color != 'black':
                parser.error('Resizing options (resize-mode, padding-color) require --size to be specified.')

        # Additional validation for background removal flags
        if (args.remove_background or args.remove_bg_full) and not args.crop_by_class:
            parser.error('--remove-background and --remove-bg-full require --crop-by-class to be set.')

        # Ensure that if --custom-output-dir is set, --outputs-base-dir is not used
        if args.custom_output_dir and args.outputs_base_dir:
            parser.error('Cannot specify both --outputs-base-dir and --custom-output-dir. Choose one.')

    # Execute the subcommand
    if args.command == 'segment':
        from wing_segmenter.segmenter import Segmenter

        segmenter = Segmenter(args)
        segmenter.process_dataset()

    elif args.command == 'scan-runs':
        from wing_segmenter.run_scanner import scan_runs

        scan_runs(dataset_path=args.dataset, output_base_dir=args.output_dir)
