import argparse
from argparse import RawTextHelpFormatter

def main():
    parser = argparse.ArgumentParser(
        prog='wingseg',
        description="Wing Segmenter CLI",
        formatter_class=RawTextHelpFormatter, # Custom control over help message formatting
    )

    subparsers = parser.add_subparsers(
        title='Commands', 
        dest='command', 
        required=True)

    # Subcommand: segment
    segment_string = 'Segment images and store segmentation masks.'
    segment_parser = subparsers.add_parser('segment',
        description=segment_string,
        help=segment_string,
        formatter_class=RawTextHelpFormatter)

    # Required argument
    segment_parser.add_argument('--dataset', 
        required=True, 
        help='''Path to dataset images.
(default: %(default)s)''')

    # Resizing options
    resize_group = segment_parser.add_argument_group('Resizing Options')

    # Dimension specifications
    resize_group.add_argument('--size', 
        nargs='+', 
        type=int,
        help='''Target size. Provide one value for square dimensions or two for width and height. 
(default: %(default)s)''')

    # Resizing mode
    resize_group.add_argument('--resize-mode', 
        choices=['distort', 'pad'], 
        default=None,
        help='''Resizing mode. "distort" resizes without preserving aspect ratio, "pad" preserves aspect ratio and adds padding if necessary. 
Required with --size. 
(default: %(default)s)''')

    # Padding options (to preserve aspect ratio)
    resize_group.add_argument('--padding-color', 
        choices=['black', 'white'], 
        default=None,
        help='''Padding color to use when --resize-mode is "pad".
(default: %(default)s)''')

    # Interpolation options
    resize_group.add_argument('--interpolation', 
        choices=['nearest', 'linear', 'cubic', 'area', 'lanczos4', 'linear_exact', 'nearest_exact'],
        default='area',
        help='''Interpolation method to use when resizing. For upscaling, "lanczos4" is recommended.
(default: %(default)s)''')

    # Bounding box padding option
    bbox_group = segment_parser.add_argument_group('Bounding Box Options')
    bbox_group.add_argument('--bbox-padding', 
        type=int, 
        default=None,
        help='''Padding to add to bounding boxes in pixels.
(default: %(default)s)''')

    # Output options within mutually exclusive group
    output_group = segment_parser.add_mutually_exclusive_group()
    output_group.add_argument('--outputs-base-dir', 
        default=None, 
        help='''Base path to store outputs under an auto-generated directory, useful for testing and managing multiple runs.
Compatible with the scan-runs command.
(default: %(default)s)''')

    output_group.add_argument('--custom-output-dir', 
        default=None, 
        help='''Fully custom directory to store all output files for a single run.
Not compatible with the scan-runs command.
(default: %(default)s)''')

    # General processing options
    segment_parser.add_argument('--sam-model', 
        default='facebook/sam-vit-base',
        help='''SAM model to use (e.g., facebook/sam-vit-base).
(default: %(default)s)''')

    segment_parser.add_argument('--yolo-model', 
        default='imageomics/butterfly_segmentation_yolo_v8:yolov8m_shear_10.0_scale_0.5_translate_0.1_fliplr_0.0_best.pt',
        help='''YOLO model to use (local path or Hugging Face repo). 
(default: %(default)s)''')

    segment_parser.add_argument('--device', 
        choices=['cpu', 'cuda'], 
        default='cpu',
        help='''Device to use for processing.
(default: %(default)s)''')

    segment_parser.add_argument('--visualize-segmentation', 
        action='store_true',
        help='''Generate and save segmentation visualizations.
(default: %(default)s)''')

    segment_parser.add_argument('--crop-by-class', 
        action='store_true',
        help='''Enable cropping of segmented classes into crops/ directory.
(default: %(default)s)''')

    segment_parser.add_argument('--force', 
        action='store_true',
        help='''Force reprocessing even if outputs already exist.
(default: %(default)s)''')

    # Background removal options
    bg_group = segment_parser.add_argument_group('Background Removal Options')

    bg_group.add_argument('--remove-crops-background', 
        action='store_true',
        help='''Remove background from cropped images.
(default: %(default)s)''')

    bg_group.add_argument('--remove-full-background', 
        action='store_true',
        help='''Remove background from the entire (resized or original) image.
(default: %(default)s)''')

    bg_group.add_argument('--background-color', 
        choices=['white', 'black'], 
        default=None,
        help='''Background color to use when removing background.
(default: %(default)s)''')
    
    # Subcommand: scan-runs
    scan_parser_string = '''List existing processing runs for a dataset. 
Requires outputs to have been generated with the --outputs-base-dir option.'''
    scan_parser = subparsers.add_parser('scan-runs', 
        description=scan_parser_string,
        help=scan_parser_string,
        formatter_class=RawTextHelpFormatter)

    scan_parser.add_argument('--dataset', 
        required=True,
        help='''Path to the dataset directory
(default: %(default)s)''')

    scan_parser.add_argument('--outputs-base-dir', 
        default=None,
        help='''Base path where outputs were stored.
(default: %(default)s)''')

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

        scan_runs(dataset_path=args.dataset, output_base_dir=args.outputs_base_dir)
