import argparse
import logging
import os

logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(
        prog='ws',
        description="Wing Segmenter CLI",
    )
    
    subparsers = parser.add_subparsers(title='Commands', dest='command', required=True)

    resize_parser = subparsers.add_parser('resize', help='Resize images and store them in a structured output directory.', formatter_class=argparse.RawTextHelpFormatter)

    resize_parser.add_argument('--source', required=True, help='Path to source images')
    resize_parser.add_argument('--output',
                            help='Base path to output resized images.\n'
                                    'Final path will include <WIDTH>x<HEIGHT>/<INTERPOLATION>.\n'
                                    'Default: <SOURCE>_resize/<WIDTH>x<HEIGHT>/<INTERPOLATION>, neighboring SOURCE.\n'
                                    'If SOURCE has nested directories, the output will mirror the structure.')
    resize_parser.add_argument('--resize-dim', nargs=2, type=int, required=True, metavar=('WIDTH', 'HEIGHT'),
                               help='Resize dimensions (WIDTH HEIGHT)')
    resize_parser.add_argument('--num-workers', type=int, default=1,
                               help='Number of parallel workers (default: 1)')
    resize_parser.add_argument('--interpolation', choices=['nearest', 'linear', 'cubic', 'area', 'lanczos4', 'linear_exact', 'nearest_exact'],
                               default='area', 
                               help='OpenCV interpolation method to use (default: area)')

    # TODO: add segmentation command

    args = parser.parse_args()

    if args.command == 'resize':
        from wing_segmenter.resize import resize_images

        # Determine output directory based on source directory
        if not args.output:
            source_dir_name = os.path.basename(os.path.normpath(args.source))
            parent_dir = os.path.dirname(os.path.abspath(args.source))
            base_output_dir = os.path.join(parent_dir, f"{source_dir_name}_resize")
        else:
            # Custom output path specified by user
            base_output_dir = args.output
        
        # Append <WIDTH>x<HEIGHT>/<INTERPOLATION> to the output path
        args.output = os.path.join(base_output_dir, f'{args.resize_dim[0]}x{args.resize_dim[1]}', args.interpolation)

        resize_images(args.source, args.output, args.resize_dim, args.num_workers, args.interpolation)


if __name__ == '__main__':
    main()
