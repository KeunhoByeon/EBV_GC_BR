import os

import cv2
import numpy as np
import openslide
from tqdm import tqdm

from tools.utils import os_walk

svs_dir = '/media/kwaklab_103/sda/data/raw_data/KBSMC/gastric/gastric_wsi/Gastric_WSI_EBV/image/'
mask_dir = '/media/kwaklab_103/sda/data/raw_data/KBSMC/gastric/gastric_wsi/Gastric_WSI_EBV/label/label_downsample8/'
save_dir = '/media/kwaklab_103/sda/data/thumbnail/KBSMC/'


def get_svs_paths():
    svs_paths = {}
    for svs_path in os_walk(svs_dir, 'svs'):
        filename = os.path.basename(svs_path)
        file_index = filename.split('.')[0].lower()
        svs_paths[file_index] = svs_path

    return svs_paths


def get_mask_paths():
    mask_paths = {}
    for mask_path in os_walk(mask_dir):
        filename = os.path.basename(mask_path)
        file_index = filename.split('.')[0].split('_')[0].lower()
        mask_paths[file_index] = mask_path

    return mask_paths


def extract_thumbnail(svs_path, mask_path):
    mask_shape = cv2.imread(mask_path).shape
    thumbnail_size = (mask_shape[1], mask_shape[0])

    slide = openslide.OpenSlide(svs_path)

    thumbnail = slide.get_thumbnail(thumbnail_size)
    thumbnail = np.array(thumbnail)
    thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2BGR)

    cv2.imwrite(os.path.join(save_dir, "{}.png".format(file_index)), thumbnail)


if __name__ == '__main__':
    os.makedirs(save_dir, exist_ok=True)

    svs_paths = get_svs_paths()
    mask_paths = get_mask_paths()

    data_paths = {}
    for file_index in svs_paths.keys():
        if file_index not in mask_paths.keys():
            continue
        data_paths[file_index] = [svs_paths[file_index], mask_paths[file_index]]

    # Parallel(n_jobs=12)(delayed(extract_thumbnail)(svs_path, mask_path) for file_index, (svs_path, mask_path) in tqdm(data_paths.items()))
    for file_index, (svs_path, mask_path) in tqdm(data_paths.items()):
        extract_thumbnail(svs_path, mask_path)
