import os

import cv2
import openslide
from joblib import Parallel, delayed
from tiatoolbox.tools.stainnorm import MacenkoNormalizer
from tqdm import tqdm

from tools.utils import os_walk

target_version = 1
n_jobs = 12
patch_size = 1024

input_dir = '/media/kwaklab_103/sdb/data/patch_data/TCGA_Stomach_452'
svs_dir = '/media/kwaklab_103/sda/data/raw_data/TCGA_Stomach_452/'
thumbnail_dir = '/media/kwaklab_103/sda/data/thumbnail/TCGA_Stomach_452_normalized_{}'.format(target_version)
output_dir = '/media/kwaklab_103/sda/data/patch_data/TCGA_Stomach_452_normalized_with_thumbnail_{}'.format(target_version)


def make_stain_normalized(data_dict):
    svs_path = data_dict['svs_path']
    thumbnail_path = data_dict['thumbnail_path']

    slide = openslide.OpenSlide(svs_path)
    w_pixels, h_pixels = slide.level_dimensions[0]

    thumbnail = cv2.imread(thumbnail_path)
    thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)

    for patch_path in data_dict['patches']:
        try:
            patch = cv2.imread(patch_path)
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

            coords = patch_path.split('_patch_')[-1].split('_class_')[0].split('_')
            x, y = int(coords[0]), int(coords[1])
            thumbnail_x_1 = int(x * thumbnail.shape[1] / w_pixels)
            thumbnail_x_2 = int((x + patch_size) * thumbnail.shape[1] / w_pixels)
            thumbnail_y_1 = int(y * thumbnail.shape[0] / h_pixels)
            thumbnail_y_2 = int((y + patch_size) * thumbnail.shape[0] / h_pixels)
            thumbnail_patch = thumbnail[thumbnail_y_1:thumbnail_y_2, thumbnail_x_1:thumbnail_x_2]

            normalizer = MacenkoNormalizer()
            normalizer.fit(thumbnail_patch)
            patch_transformed = normalizer.transform(patch)

            # if True:
            #     output = np.hstack([cv2.resize(patch, (512, 512)), cv2.resize(thumbnail_patch, (512, 512)), cv2.resize(patch_transformed, (512, 512))])
            #     output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            #     cv2.imshow('T', output)
            #     cv2.waitKey(0)

            save_path = patch_path.replace(input_dir, output_dir)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            patch_transformed = cv2.cvtColor(patch_transformed, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, patch_transformed)

        except Exception as e:
            with open(os.path.join(output_dir, 'failed.txt'), 'a') as wf:
                wf.write(patch_path + '\n')
            print(patch_path, e)

            patch = cv2.imread(patch_path)
            save_path = patch_path.replace(input_dir, output_dir)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, patch)


if __name__ == "__main__":
    svs_paths = {}
    for svs_path in os_walk(svs_dir, '.svs'):
        filename = os.path.basename(svs_path)
        file_full_index = filename.replace(os.path.splitext(filename)[-1], '').lower()
        svs_paths[file_full_index] = svs_path

    thumbnail_paths = {}
    for thumbnail_path in os_walk(thumbnail_dir, 'images'):
        filename = os.path.basename(thumbnail_path)
        file_full_index = filename.replace(os.path.splitext(filename)[-1], '').lower()
        thumbnail_paths[file_full_index] = thumbnail_path

    input_data = {}
    for patch_path in os_walk(input_dir, 'images'):
        file_full_index = os.path.basename(patch_path).split('_patch')[0].lower()
        file_index = '-'.join(file_full_index.split('-')[:3])
        if file_full_index not in input_data.keys():
            if file_full_index not in svs_paths or file_full_index not in thumbnail_paths:
                continue
            input_data[file_full_index] = {'patches': [], 'svs_path': svs_paths[file_full_index], 'thumbnail_path': thumbnail_paths[file_full_index]}
        input_data[file_full_index]['patches'].append(patch_path)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'failed.txt'), 'w') as wf:
        pass

    if n_jobs > 1:
        Parallel(n_jobs=n_jobs)(delayed(make_stain_normalized)(data_dict) for file_full_index, data_dict in tqdm(input_data.items()))
    else:
        for _, data_dict in tqdm(input_data.items()):
            make_stain_normalized(data_dict)
