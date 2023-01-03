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
        img_paths.append(img_path)
    img_paths.sort()
    random.shuffle(img_paths)
    line_size = int(len(img_paths) ** 0.5)
    img_paths = img_paths[:line_size ** 2]

    total, line = [], []
    for img_path in tqdm(img_paths):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (512, 512))
        y, x = img.shape[0] // 2, img.shape[1] // 2
        img = img[y - 4: y + 4, x - 4: x + 4]

        line.append(img)
        if len(line) >= line_size:
            total.append(np.hstack(line))
            del line
            line = []
    output = np.vstack(total)

    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(os.path.join(save_dir, 'target_2.png'), output)

    # output = cv2.resize(output, (512, 512))
    # cv2.imshow('T', output)
    # cv2.waitKey(0)
