import os
import cv2
import numpy as np


def stereo_disparity_map(left_img_path, right_img_path, index):
    left_img = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)

    stereo_img = cv2.StereoBM_create(numDisparities=32, blockSize=7)

    disparity_map = stereo_img.compute(left_img, right_img).astype(np.float32)

    cv2.imwrite(os.path.join(str(save_root), f'{index}.png'), disparity_map * 255)


def resize_image(img_path):
    image = cv2.imread(img_path).astype(np.float32)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    resized_image = cv2.resize(image, dsize=(1633, 1225))

    return resized_image


save_root = "./inputs"

if __name__ == '__main__':
    left_path = 'inputs/47.jpg'
    right_path = './stereo_data/47_right.jpg'

    # stereo_disparity_map(left_path, right_path, 47)
    # cv2.imwrite(os.path.join(save_root, '47_2.jpg'), resize_image('inputs/47.jpg'))

    # ii = cv2.imread('./inputs/47_2.jpg').astype(np.float32)

    # print(ii.shape)
