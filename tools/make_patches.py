import multiprocessing as mp
import os

import cv2
import numpy as np
import openslide
from joblib import Parallel, delayed
from tqdm import tqdm

from utils import load_annotation, os_walk

MEAN_THRESH = 240.
MASK_THRESH = 0.


def make_patches(file_full_index, data_dict, read_size=1024, step=1.0):
    svs_path = data_dict['svs_path']
    mask_path = data_dict['mask_path']
    patch_save_dir = data_dict['patch_save_dir']
    label = data_dict['label']

    mask = None
    if mask_path is not None:
        # TODO: I think I should check mask shape before I code it
        pass

    slide = openslide.OpenSlide(svs_path)
    w_pixels, h_pixels = slide.level_dimensions[0]

    os.makedirs(save_dir, exist_ok=True)
    for w_i in range(0, w_pixels, int(read_size * step)):
        for h_i in range(0, h_pixels, int(read_size * step)):
            if w_i + read_size > w_pixels > read_size:
                w_i = w_pixels - read_size
            if h_i + read_size > h_pixels > read_size:
                h_i = h_pixels - read_size

            slide_img = slide.read_region((w_i, h_i), 0, (read_size, read_size))
            slide_img = np.array(slide_img)

            if mask is not None:
                mask_patch = mask[h_i:h_i + read_size, w_i: w_i + read_size]
                if np.mean(mask_patch) <= MASK_THRESH:
                    continue
            elif np.mean(slide_img) > MEAN_THRESH:
                continue

            save_path = os.path.join(patch_save_dir, '{}_patch_{}_{}_class_{}.png'.format(file_full_index, w_i, h_i, label))
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, slide_img)


if __name__ == '__main__':
    xlsx_path = '/media/kwaklab_103/sda/data/raw_data/TCGA_Stomach_452/STAD_molecular_subtype TCGA data.xlsx'
    svs_dir = '/media/kwaklab_103/sda/data/raw_data/TCGA_Stomach_452/'
    mask_dir = '/media/kwaklab_103/sda/data/raw_data/TCGA_Stomach_452_tissue_mask'
    save_dir = '/media/kwaklab_103/sdb/data/patch_data/TCGA_Stomach_452/'

    anno_data = load_annotation(xlsx_path)

    svs_data = {}
    for svs_path in os_walk(svs_dir, '.svs'):
        filename = os.path.basename(svs_path)
        file_full_index = filename.replace(os.path.splitext(filename)[-1], '').lower()
        file_index = '-'.join(file_full_index.split('-')[:3])
        if file_index in anno_data.keys():
            label = anno_data[file_index]
            label = 2 if label == 0 else label
        else:
            label = 0
        patch_save_dir = os.path.dirname(svs_path).replace(svs_dir, save_dir)
        svs_data[file_full_index] = {'svs_path': svs_path, 'patch_save_dir': patch_save_dir, 'mask_path': None, 'label': label}

    for mask_path in os_walk(mask_dir, ('.png', '.jpg', '.jpeg')):
        filename = os.path.basename(mask_path)
        file_full_index = filename.replace(os.path.splitext(filename)[-1], '').lower()
        svs_data[file_full_index]['mask_path'] = mask_path

    Parallel(n_jobs=mp.cpu_count() - 1)(delayed(make_patches)(file_full_index, data_dict) for file_full_index, data_dict in tqdm(svs_data.items()))
