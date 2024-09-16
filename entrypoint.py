import yaml
import subprocess
import os

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def execute_command(command, flags):
    command_line = [command] + [f"--{key} {value}" for key, value in flags.items()]
    command_str = ' '.join(command_line)
    print(f"Executing command: {command_str}")
    subprocess.run(command_str, shell=True, check=True)

def main():
    # Load configuration from YAML file
    config_file = os.getenv('CONFIG_FILE', '/config/wing_segmentation_config.yaml')
    config = load_config(config_file)

    # Extract command and flags from the config
    command = config.get('command')
    flags = config.get('flags', {})

    # Map command to the appropriate script
    command_mapping = {
        "resize_images_flat_dir": "/usr/src/app/preprocessing_scripts/resize_images_flat_dir.py",
        "resize_images_subfolders": "/usr/src/app/preprocessing_scripts/resize_images_subfolders.py",
        "yolo_sam_predict_mask": "/usr/src/app/segmentation_scripts/yolo_sam_predict_mask.py",
        "remove_background_black": "/usr/src/app/segmentation_scripts/remove_background_black.py",
        "remove_background_white": "/usr/src/app/segmentation_scripts/remove_background_white.py",
        "select_wings": "/usr/src/app/segmentation_scripts/select_wings.py",
        "crop_wings_out": "/usr/src/app/segmentation_scripts/crop_wings_out.py",
        "create_wing_folders": "/usr/src/app/landmark_scripts/create_wing_folders.py",
        "flip_images_horizontally": "/usr/src/app/landmark_scripts/flip_images_horizontally.py",
    }

    if command in command_mapping:
        script_path = command_mapping[command]
        execute_command(f"python3 {script_path}", flags)
    else:
        print(f"Unknown command: {command}")
        exit(1)

if __name__ == "__main__":
    main()
