# Lepidoptera Wing Segmentation

This repository contains the scripts necessary to extract the following components using a YOLO v8 object detector and Meta's Segment Anything (SAM) model from butterfly images:
- wings (right forewing, right hindwing, left forewing, left hindwing) 
- ruler
- metadata label
- color palette


## Setting up your environment

1. Create conda environment
```
conda create -n yolo_segmentation python=3.10
```

2. Activate environment
```
conda activate yolo_segmentation
```

3. Install requirements
```
pip install -r requirements.txt
```

## 1. Preprocessing (OPTIONAL. Skip to Step 2 if you don't wish to resize your images)

The YOLO model does not require that you resize your images. However, if you wish to resize your images regardless, you can follow the steps below to do so. 

To resize your images, you can use either the `resize_images_flat_dir.py` or the `resize_images_subfolders.py` file in the `preprocessing_scripts` folder. The only difference between the two is the assumed folder structures. Both scripts will create a new directory containing your resized images such that the original images are not overwritten/modified. 

### Option 1: Resizing with `resize_images_flat_dir.py`
command: 
```
python3 resize_images_flat_dir.py --source /path/to/source/image/dataset/folder --output /path/to/new/folder/to/store/images --resize_dim 256 256
```

The script `resize_images_flat_dir.py` expects your source folder structure to look as such:
```
|-- Source_Image_Folder
|   |-- image_1.jpg
|   |-- image_2.jpg
|   |-- image_3.jpg
|   |-- image_4.jpg
|   |-- image_5.jpg
```

### Option 2: Resizing with `resize_images_subfolders.py`
command: 
```
python3 resize_images_subfolders.py --source /path/to/source/image/dataset/folder --output /path/to/new/folder/to/store/images --resize_dim 256 256
```

The script `resize_images_subfolders.py` expects your source folder structure to look as such:

```
|-- Source_Image_Folder
|   |--species_folder_1
|   |   |-- image_1.jpg
|   |   |-- image_2.jpg
|   |--species_folder_2
|   |   |-- image_1.jpg
|   |   |-- image_2.jpg
|   |--species_folder_3
|   |   |-- image_1.jpg
|   |   |-- image_2.jpg

```

## 2. Getting Segmentation Masks using YOLO Detection + SAM Model

To obtain masks for your set of images, run the `yolo_sam_predict_mask.py` script in the `segmentation_scripts` folder. The result will be a new folder containing all the segmentation masks for each of your images in the input directory.

Command: 

```
python3 wing-segmentation/segmentation_scripts/yolo_predict_masks.py --dataset /path/to/your/images --mask_csv /path/where/to/store/segmentation_info.csv
```

Arguments explained: 

`--dataset` is the full path to your folder containing your images you wish to obtain masks for. (Example: /User/micheller/data/jiggins_256_256)

`--mask_csv` is the path location at which you want to store the csv that gets created detailing which segmentation categories exist in the mask generated for each image. (Optional. Default segmentation.csv will be saved in the same directory from where you run this script.)

## 3. Removing background and background items from your wing images

The `segmentation_scripts` folder contains python scripts to help you remove the background of your images using the segmentation masks from our YOLO-SAM model. 

**Remove background only and replace with black background**

```
python3 wing-segmentation/segmentation_scripts/remove_background_black.py --image_dataset_path <path> --mask_dataset_path <path> --main_folder_name <folder_name>
```

**Remove background only and replace with white background**

```
python3 wing-segmentation/segmentation_scripts/remove_background_white.py --image_dataset_path <path> --mask_dataset_path <path> --main_folder_name <folder_name>
```

**Remove background and all items that are not wings. Wings are placed against a white background**

```
python3 wing-segmentation/segmentation_scripts/select_wings.py --images <path> --masks <path> --main_folder <folder_name>
```

## 4. Using Segmentation Masks to Extract Wings from Images

After obtaining masks for our images, we can crop out the forewings and hindwings by running the following `crop_wings_out.py` script in the `segmentation_scripts` folder:

Command:

```
python3 wing-segmentation/segmentation_scripts/crop_wings_out.py --images /path/to/butterfly/images --masks /path/to/segmentation/masks --output_folder /path/to/save/cropped/wings/to --pad <pixels to extend crop window by>
```

The `crop_wings_out.py` file will produce images like those below:

<img src="readme_images/CAM016015_d_lfw.png" alt="Alt text" width="100" height="100">
<img src="readme_images/CAM016015_d_rfw.png" alt="Alt text" width="100" height="100">
<img src="readme_images/CAM016022_d_lhw.png" alt="Alt text" width="100" height="100">
<img src="readme_images/CAM016019_d_rhw.png" alt="Alt text" width="100" height="100">


Arguments explained: 

`--images` is the path to the folder containing the set of images we got masks for in step 2.

`--masks` is the path to the folder created during step 2 containing the segmentation masks for the images.

`--output_folder` is the name of the folder you want to give to the folder that will contain the cropped wings.

`--pad` is the number of pixels to use as padding and extend the crop window by when cropping out wings. This argument is optional with a default value of 50. If the individual cropped wings are including neighboring wings too much for your liking, reduce this number to get a tighter window/crop around the wing. If the individual wing is getting cut off in the crop, increase this number. 

The cropped wing images will be named in this structure: `<original name>_wing_#.png`

The number following `wing` can be mapped as follows:

`1`: right forewing

`2`: left forewing

`3`: right hindwing

`4`: left hindwing

## 5. Creating seperate wing folders and flipping wings

The `landmark_scripts` folder contains python scripts to sort cropped wings into wing folders and flip images horizontally if needed.

Commands:

**Create wing folders**
```
python3 wing-segmentation/landmark_scripts/create_wing_folders.py --input_dir /path/to/folder/where/we/store/cropped/wing/results
```

**Flip images**
```
python3 wing-segmentation/landmark_scripts/flip_images_horizontally.py --input_dir /path/to/wing/category/folder
```

# CLI Development
To use and continue building the CLI features:

1. Set up an environment and install the package.
```bash
conda create -n ws -c conda-forge --solver=libmamba python=3.10 uv -y
```
```bash
conda activate ws
```
```bash
uv pip install -e .[dev]
```
> Note: [`uv`](https://github.com/astral-sh/uv) is a fast, Rust-based package manager.


