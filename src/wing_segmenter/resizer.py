import cv2
from wing_segmenter.constants import INTERPOLATION_METHODS

def get_interpolation_method(interpolation):
    return INTERPOLATION_METHODS.get(interpolation, cv2.INTER_AREA)

def resize_image(image, size, resize_mode, padding_color, interpolation):
    """
    Resizes the image based on the resize mode, dimensions, and padding settings.
    The image will always be resized to the target dimensions.
    """
    original_height, original_width = image.shape[:2]

    # If no resize dimensions are specified, return the original image
    if not size:
        return image

    # Determine target dimensions
    if len(size) == 1:
        target_width = target_height = size[0]
    else:
        target_width, target_height = size

    if resize_mode == 'distort':
        # Resize without preserving AR
        resized_image = cv2.resize(image, (target_width, target_height), interpolation=get_interpolation_method(interpolation))
        return resized_image

    elif resize_mode == 'pad':
        # Compute scaling factor while preserving AR
        scale = min(target_width / original_width, target_height / original_height)

        # Compute new dimensions after scaling
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)

        # Resize image to fit within target dimensions while preserving AR
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=get_interpolation_method(interpolation))

        # Calculate padding to fit the target dimensions
        delta_w = target_width - new_width
        delta_h = target_height - new_height
        top = delta_h // 2
        bottom = delta_h - top
        left = delta_w // 2
        right = delta_w - left

        # Define padding color
        if padding_color == 'black':
            pad_color = [0, 0, 0]
        else:
            pad_color = [255, 255, 255]

        # Add padding to match the target size
        padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)
        return padded_image

    else:
        # Fallback to returning the original image if no valid resize mode is specified
        return image
