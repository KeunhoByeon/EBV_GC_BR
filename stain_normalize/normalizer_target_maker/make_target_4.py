import os

import cv2
import numpy as np
from tqdm import tqdm

from tools.utils import os_walk

thumbnail_dir = '/media/kwaklab_103/sda/data/thumbnail/KBSMC/'
mask_dir = '/media/kwaklab_103/sda/data/raw_data/KBSMC/gastric/gastric_wsi/Gastric_WSI_EBV/label/label_downsample8/'
save_dir = '../data'


def get_thumbnail_paths():
    thumbnail_paths = {}
    for thumbnail_path in os_walk(thumbnail_dir):
        filename = os.path.basename(thumbnail_path)
        file_index = filename.split('.')[0].lower()
        thumbnail_paths[file_index] = thumbnail_path

    return thumbnail_paths


def get_mask_paths():
    mask_paths = {}
    for mask_path in os_walk(mask_dir):
        filename = os.path.basename(mask_path)
        file_index = filename.split('.')[0].split('_')[0].lower()
        mask_paths[file_index] = mask_path

    return mask_paths


if __name__ == '__main__':
    thumbnail_paths = get_thumbnail_paths()
    mask_paths = get_mask_paths()

    output_list = []
    output_lists = []
    for file_index in tqdm(thumbnail_paths.keys()):
        if file_index not in mask_paths.keys():
            continue

        thumbnail_path = thumbnail_paths[file_index]
        mask_path = mask_paths[file_index]

        thumbnail = cv2.imread(thumbnail_path)
        thumbnail = cv2.resize(thumbnail, (1024, 512))
        output_list.append(thumbnail)

        if len(output_list) >= 3:
            output_lists.append(np.hstack(output_list))
            output_list = []
    output = np.vstack(output_lists)

    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(os.path.join(save_dir, 'target_4.png'), output)

    # cv2.imshow('T', output)
    # cv2.waitKey(0)
