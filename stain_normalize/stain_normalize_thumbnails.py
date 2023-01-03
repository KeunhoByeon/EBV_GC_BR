import os

import cv2
import numpy as np
from joblib import Parallel, delayed
from tiatoolbox.tools.stainnorm import MacenkoNormalizer
from tqdm import tqdm

from tools.utils import os_walk

target_version = 1
n_jobs = 12

target_path = "./data/target_{}.png".format(target_version)
input_dir = '/media/kwaklab_103/sda/data/thumbnail/TCGA_Stomach_452'
mask_dir = '/media/kwaklab_103/sda/data/raw_data/TCGA_Stomach_452_tissue_mask'
output_dir = '/media/kwaklab_103/sda/data/thumbnail/TCGA_Stomach_452_normalized_{}'.format(target_version)


def make_stain_normalized(data_dict, normalizer):
    try:
        source_path = data_dict['thumbnail']
        mask_path = data_dict['mask']

        save_path = source_path.replace(input_dir, output_dir)

        source = cv2.imread(source_path, cv2.IMREAD_COLOR)
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        mask = cv2.resize(mask, (source.shape[1], source.shape[0]))

        source = np.where(mask > 127, source, 255)

        transformed = normalizer.transform(source)
        transformed = np.where(mask > 127, transformed, 255).astype(np.uint8)

        transformed = cv2.cvtColor(transformed, cv2.COLOR_RGB2BGR)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, transformed)
    except Exception as e:
        source_path = data_dict['thumbnail']
        save_path = source_path.replace(input_dir, output_dir)
        print("Error {}".format(source_path))

        source = cv2.imread(source_path, cv2.IMREAD_COLOR)
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)

        transformed = normalizer.transform(source)
        transformed = cv2.cvtColor(transformed, cv2.COLOR_RGB2BGR)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, transformed)


if __name__ == "__main__":
    target = cv2.imread(target_path)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

    mask_data = {}
    for mask_path in os_walk(mask_dir, ('.png', '.jpg', '.jpeg')):
        filename = os.path.basename(mask_path)
        file_full_index = filename.replace(os.path.splitext(filename)[-1], '').lower()
        mask_data[file_full_index] = mask_path

    data = {}
    for thunmbnail_path in os_walk(input_dir, ('.png', '.jpg', '.jpeg')):
        filename = os.path.basename(thunmbnail_path)
        file_full_index = filename.replace(os.path.splitext(filename)[-1], '').lower()
        if file_full_index not in mask_data.keys():
            continue
        data[file_full_index] = {'thumbnail': thunmbnail_path, 'mask': mask_data[file_full_index]}

    normalizer = MacenkoNormalizer()
    normalizer.fit(target)

    if n_jobs > 1:
        Parallel(n_jobs=n_jobs)(delayed(make_stain_normalized)(data_dict, normalizer) for _, data_dict in tqdm(data.items()))
    else:
        for _, data_dict in tqdm(data.items()):
            make_stain_normalized(data_dict, normalizer)
