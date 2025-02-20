import os
import json
import glob
from rich.table import Table
from rich.console import Console

def scan_runs(dataset_path, output_base_dir=None, custom_output_dir=None):
    dataset_path = os.path.abspath(dataset_path)

    if not os.path.exists(dataset_path):
        console = Console()
        console.print(f"[red]Error: The dataset path '{dataset_path}' does not exist.[/red]")
        return

    # Determine the base directory to search for runs
    dataset_name = os.path.basename(dataset_path.rstrip('/\\'))
    
    if custom_output_dir:
        # If a custom output directory is provided, scan only that directory without expecting specific naming
        run_dirs = [custom_output_dir] if os.path.exists(custom_output_dir) else []
    else:
        if output_base_dir:
            output_dir = os.path.abspath(output_base_dir)
        else:
            output_dir = os.path.dirname(dataset_path)

        # Search for run directories in the specified output directory
        pattern = f"{output_dir}/{dataset_name}_*"
        run_dirs = glob.glob(pattern)

    console = Console()

    if not run_dirs:
        if custom_output_dir:
            console.print(f"[red]No processing runs found in the custom output directory '{custom_output_dir}' for dataset '{dataset_name}'.[/red]")
        elif output_base_dir:
            console.print(f"[red]No processing runs found in '{output_dir}' for dataset '{dataset_name}'.[/red]")
        else:
            console.print(f"[red]No processing runs found for dataset '{dataset_name}' in default location ('{output_dir}').[/red]")
        return

    console.print(f"[bold green]Found {len(run_dirs)} processing runs for dataset '{dataset_name}':[/bold green]\n")

    table = Table(title="Processing Runs", header_style="", show_lines=True)

    table.add_column("Run #", justify="right", no_wrap=True, width=5)
    table.add_column("Run UUID Prefix", justify="left", no_wrap=False, width=8)
    table.add_column("Completed", justify="center", no_wrap=True, width=9)
    table.add_column("Num Images", justify="right", no_wrap=False, width=10)
    table.add_column("Resize Dims", justify="center", no_wrap=False, width=11)
    table.add_column("Resize Mode", justify="center", no_wrap=False, width=7)
    table.add_column("Interp", justify="center", no_wrap=True, min_width=13)
    table.add_column("BBox Pad", justify="right", no_wrap=True, min_width=8)
    table.add_column("Errors", justify="center", no_wrap=True, min_width=6)

    for idx, run_dir in enumerate(run_dirs, 1):
        metadata_path = os.path.join(run_dir, 'metadata', 'run_metadata.json')

        if not os.path.exists(metadata_path):
            table.add_row(str(idx), "Missing run_metadata.json", "", "", "", "", "", "", "")
            continue

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        completed = 'Yes' if metadata['run_status'].get('completed') else 'No'

        num_images = str(metadata['dataset'].get('num_images', 'Unknown'))
        resize_dims = metadata['run_parameters'].get('size', 'None (original)')
        resize_mode = str(metadata['run_parameters'].get('resize_mode', 'None'))

        # Handle 'size' being a list with 1 or 2 elements or other types
        if isinstance(resize_dims, list):
            if len(resize_dims) == 1:
                resize_dims_str = f"{resize_dims[0]}x{resize_dims[0]}"
            elif len(resize_dims) == 2:
                resize_dims_str = f"{resize_dims[0]}x{resize_dims[1]}"
            else:
                resize_dims_str = str(resize_dims)
        else:
            resize_dims_str = str(resize_dims)

        interpolation = str(metadata['run_parameters'].get('interpolation', 'None'))
        
        bbox_padding = str(metadata['run_parameters'].get('bbox_pad_px', 'None'))

        errors = str(metadata['run_status'].get('errors', 'None'))

        # Truncate run UUID to save table space
        run_uuid_prefix = os.path.basename(run_dir).split('_')[-1][:8] if not custom_output_dir else "CustomDir"

        table.add_row(
            str(idx),
            run_uuid_prefix,
            completed,
            num_images,
            resize_dims_str,
            resize_mode,
            interpolation,
            bbox_padding,
            errors
        )

    console.print(table)
