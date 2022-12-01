import math
import multiprocessing as mp
import os

import cv2
import numpy as np
import openslide
from joblib import Parallel, delayed
from tqdm import tqdm

from utils import load_annotation, os_walk, resize_and_pad_image

MEAN_THRESH = 240.
MASK_THRESH = 0.1

EXCEPT_NO_LABEL = True
EXCEPT_NO_MASK = True
EXCEPT_SMALL = True
EXCEPT_EXIST = True
EXCEPT_BLACK_EDGE = True

xlsx_path = '/media/kwaklab_103/sda/data/raw_data/TCGA_Stomach_452/STAD_molecular_subtype TCGA data.xlsx'
svs_dir = '/media/kwaklab_103/sda/data/raw_data/TCGA_Stomach_452/'
mask_dir = '/media/kwaklab_103/sda/data/raw_data/TCGA_Stomach_452_tissue_mask'
save_dir = '/media/kwaklab_103/sdb/data/patch_data/TCGA_Stomach_452/'


def make_patches(file_full_index, data_dict, read_size=1024, step=1.0):
    svs_path = data_dict['svs_path']
    mask_path = data_dict['mask_path']
    patch_save_dir = data_dict['patch_save_dir']
    label = data_dict['label']

    if EXCEPT_NO_LABEL and label is None:
        return
    if EXCEPT_NO_MASK and mask_path is None:
        return

    label = "None" if label is None else label
    slide = openslide.OpenSlide(svs_path)
    w_pixels, h_pixels = slide.level_dimensions[0]

    if EXCEPT_SMALL and (w_pixels < 20480 or h_pixels < 20480):
        return

    if mask_path is not None:
        mask_ratio = min(1024 / h_pixels, 1024 / w_pixels)
        mask_shape = (int(h_pixels * mask_ratio), int(w_pixels * mask_ratio))
        tissue_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        tissue_mask = resize_and_pad_image(tissue_mask, target_size=mask_shape, padding=False)
        tissue_mask = tissue_mask.astype(float)
        tissue_mask /= 255.

    os.makedirs(save_dir, exist_ok=True)
    for w_i in range(0, w_pixels, int(read_size * step)):
        for h_i in range(0, h_pixels, int(read_size * step)):
            save_path = os.path.join(patch_save_dir, '{}_patch_{}_{}_class_{}.png'.format(file_full_index, w_i, h_i, label))
            if EXCEPT_EXIST and os.path.isfile(save_path):
                continue

            # Mask
            if MASK_THRESH is not None and MASK_THRESH > 0 and mask_path is not None:
                coord_x_1 = w_i * mask_ratio
                coord_y_1 = h_i * mask_ratio
                coord_x_2 = coord_x_1 + read_size * mask_ratio
                coord_y_2 = coord_y_1 + read_size * mask_ratio
                if EXCEPT_BLACK_EDGE:
                    if math.floor(coord_x_1) < 0 or math.floor(coord_y_1) < 0:
                        continue
                    elif math.ceil(coord_x_2) >= tissue_mask.shape[1] or math.ceil(coord_y_2) >= tissue_mask.shape[0]:
                        continue
                coord_x_1, coord_y_1 = max(math.floor(coord_x_1), 0), max(math.floor(coord_y_1), 0)
                coord_x_2, coord_y_2 = min(math.ceil(coord_x_2), tissue_mask.shape[1]), min(math.ceil(coord_y_2), tissue_mask.shape[0])
                if np.mean(tissue_mask[coord_y_1:coord_y_2, coord_x_1:coord_x_2]) < MASK_THRESH:
                    continue

            slide_img = slide.read_region((w_i, h_i), 0, (read_size, read_size))
            slide_img = cv2.cvtColor(np.array(slide_img), cv2.COLOR_RGB2BGR)

            if MEAN_THRESH is not None:
                slide_mean = [np.mean(slide_img[:, :, 0]), np.mean(slide_img[:, :, 1]), np.mean(slide_img[:, :, 2])]
                if slide_mean[0] > MEAN_THRESH and slide_mean[1] > MEAN_THRESH and slide_mean[2] > MEAN_THRESH:
                    continue

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, slide_img)


if __name__ == '__main__':
    anno_data = load_annotation(xlsx_path)

    mask_paths = {}
    for mask_path in os_walk(mask_dir, ('.png', '.jpg', '.jpeg')):
        filename = os.path.basename(mask_path)
        file_full_index = filename.replace(os.path.splitext(filename)[-1], '').lower()
        mask_paths[file_full_index] = mask_path

    svs_data = {}
    for svs_path in os_walk(svs_dir, '.svs'):
        filename = os.path.basename(svs_path)
        file_full_index = filename.replace(os.path.splitext(filename)[-1], '').lower()
        file_index = '-'.join(file_full_index.split('-')[:3])
        if file_index.upper() in anno_data.keys():
            label = anno_data[file_index.upper()]
            label = 2 if label == 0 else label
        else:
            continue
        if file_full_index in mask_paths.keys():
            mask_path = mask_paths[file_full_index]
        else:
            continue
        patch_save_dir = os.path.dirname(svs_path).replace(svs_dir, save_dir)
        svs_data[file_full_index] = {'svs_path': svs_path, 'patch_save_dir': patch_save_dir, 'mask_path': mask_path, 'label': label}

    Parallel(n_jobs=mp.cpu_count() - 1)(delayed(make_patches)(file_full_index, data_dict) for file_full_index, data_dict in tqdm(svs_data.items()))
