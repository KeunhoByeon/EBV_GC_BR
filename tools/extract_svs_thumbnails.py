import os

import cv2
import numpy as np
import openslide
from tqdm import tqdm

from tools.utils import load_annotation

svs_dir = '/media/kwaklab_103/sda/data/raw_data/TCGA_Stomach_452/'
xlsx_path = '/media/kwaklab_103/sda/data/raw_data/TCGA_Stomach_452/STAD_molecular_subtype TCGA data.xlsx'
save_dir = '/media/kwaklab_103/sda/data/thumbnail/TCGA_Stomach_452/'


def get_svs_paths(svs_dir):
    svs_paths = []
    for path, dir, files in os.walk(svs_dir):
        for filename in files:
            if os.path.splitext(filename)[-1] != '.svs':
                continue
            svs_paths.append(os.path.join(path, filename))

    return svs_paths


def extract_thumbnail(svs_path, thumbnail_size):
    slide = openslide.OpenSlide(svs_path)
    w_pixels, h_pixels = slide.level_dimensions[0]

    ratio = min(thumbnail_size / w_pixels, thumbnail_size / h_pixels)
    thumbnail_shape = (int(w_pixels * ratio), int(h_pixels * ratio))

    thumbnail = slide.get_thumbnail(thumbnail_shape)
    thumbnail = np.array(thumbnail)
    thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2BGR)

    return thumbnail


if __name__ == '__main__':
    os.makedirs(save_dir, exist_ok=True)

    for svs_path in tqdm(get_svs_paths(svs_dir)):
        filename = os.path.basename(svs_path)
        file_full_index = filename.replace(os.path.splitext(filename)[-1], '').lower()
        file_index = '-'.join(file_full_index.split('-')[:3])

        thumbnail = extract_thumbnail(svs_path, thumbnail_size=8192)
        cv2.imwrite(os.path.join(save_dir, "{}.png".format(file_full_index)), thumbnail)
