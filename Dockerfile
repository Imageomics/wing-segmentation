FROM python:3.10

WORKDIR /usr/src/app

RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

COPY . .

RUN uv pip install --no-cache-dir -r requirements.txt

# Ensure the scripts are executable
RUN chmod +x /usr/src/app/preprocessing_scripts/*.py
RUN chmod +x /usr/src/app/segmentation_scripts/*.py
RUN chmod +x /usr/src/app/landmark_scripts/*.py

RUN echo "alias resize_images_flat='/usr/src/app/preprocessing_scripts/resize_images_flat_dir.py'" >> ~/.bashrc && \
    echo "alias resize_images_subfolders='/usr/src/app/preprocessing_scripts/resize_images_subfolders.py'" >> ~/.bashrc && \
    echo "alias get_segmentation_masks='/usr/src/app/segmentation_scripts/yolo_sam_predict_mask.py'" >> ~/.bashrc && \
    echo "alias remove_background_black='/usr/src/app/segmentation_scripts/remove_background_black.py'" >> ~/.bashrc && \
    echo "alias remove_background_white='/usr/src/app/segmentation_scripts/remove_background_white.py'" >> ~/.bashrc && \
    echo "alias select_wings='/usr/src/app/segmentation_scripts/select_wings.py'" >> ~/.bashrc && \
    echo "alias crop_wings='/usr/src/app/segmentation_scripts/crop_wings_out.py'" >> ~/.bashrc && \
    echo "alias create_wing_folders='/usr/src/app/landmark_scripts/create_wing_folders.py'" >> ~/.bashrc && \
    echo "alias flip_images='/usr/src/app/landmark_scripts/flip_images_horizontally.py'" >> ~/.bashrc

RUN echo "source ~/.bashrc" >> ~/.profile

ENTRYPOINT ["/bin/bash", "-c", "source ~/.profile && exec bash"]
