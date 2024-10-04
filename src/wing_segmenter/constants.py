import cv2

CLASSES = {
    0: 'background',
    1: 'right_forewing',
    2: 'left_forewing',
    3: 'right_hindwing',
    4: 'left_hindwing',
    5: 'ruler',
    6: 'white_balance',
    7: 'label',
    8: 'color_card',
    9: 'body'
}

INTERPOLATION_METHODS = {
    'nearest': cv2.INTER_NEAREST,
    'linear': cv2.INTER_LINEAR,
    'cubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos4': cv2.INTER_LANCZOS4,
    'linear_exact': cv2.INTER_LINEAR_EXACT,
    'nearest_exact': cv2.INTER_NEAREST_EXACT,
}
