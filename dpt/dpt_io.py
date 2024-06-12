import cv2
import numpy as np


def read_image(path):
    img = cv2.imread(path)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

    return img


def write_depth(file_name, depth, bits=1, absolute_depth=False):

    if absolute_depth:
        result = depth
    else:
        depth_min = depth.min()
        depth_max = depth.max()

        max_val = 2 ** (8 * bits) - 1

        if depth_max - depth_min > np.finfo('float').eps:
            result = max_val * (depth - depth_min) / (depth_max - depth_min)
        else:
            result = np.zeros(depth.shape, dtype=depth.dtype)

    if bits == 1:
        cv2.imwrite(file_name + '.png', result.astype(np.uint8), [cv2.IMWRITE_PNG_COMPRESSION, 0])
    elif bits == 2:
        cv2.imwrite(file_name + '.png', result.astype(np.uint16), [cv2.IMWRITE_PNG_COMPRESSION, 0])
