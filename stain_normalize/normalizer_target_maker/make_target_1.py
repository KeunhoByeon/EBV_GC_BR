import os
import random

import cv2
import numpy as np
from tqdm import tqdm

from tools.utils import os_walk

base_dir = '/media/kwaklab_103/sda/data/patch_data/KBSMC/gastric/gastric_EBV_1024'
save_dir = '../data'

if __name__ == '__main__':
    random.seed(103)
    img_paths = []
    for img_path in os_walk(base_dir, ext=['png', 'jpg', 'jpeg']):
        label = int(os.path.basename(img_path).split('.')[0].split('_class_')[1])
        if label == 1:
            img_paths.append(img_path)
    print(len(img_paths))

    img_paths.sort()
    random.shuffle(img_paths)
    img_paths = img_paths[:64 * 64]

    img_paths = np.array(img_paths)
    img_paths = img_paths.reshape((64, 64))

    imgs = []
    for line in tqdm(img_paths):
        imgs_line = [cv2.resize(cv2.imread(img_path), (64, 64)) for img_path in line]
        imgs_line = np.hstack(imgs_line)
        imgs.append(imgs_line)
    output = np.vstack(imgs)

    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(os.path.join(save_dir, 'target_1.png'), output)

    # cv2.imshow('T', output)
    # cv2.waitKey(0)
