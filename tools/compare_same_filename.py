import os

import cv2
import numpy as np

from tools.utils import os_walk

# compare_dirs = [
#     '/media/kwaklab_103/sdb/data/patch_data/TCGA_Stomach_452',
#     '/media/kwaklab_103/sdc/data/patch_data/TCGA_Stomach_452_stain_normalize_with_thumbnail',
#     '/media/kwaklab_103/sdc/data/patch_data/TCGA_Stomach_452_stain_normalize_custom_ver1',
# ]
# compare_dirs = [
#     '/home/kwaklab_103/lib/EBV_GC_BR/results/20221201192712_base_model/eval_TCGA/70',
#     '/home/kwaklab_103/lib/EBV_GC_BR/results/20221201192712_base_model/eval_TCGA_stain_normalize/70',
#     '/home/kwaklab_103/lib/EBV_GC_BR/results/20221201192712_base_model/eval_TCGA_stain_normalize_custom_ver2/70',
#     '/home/kwaklab_103/lib/EBV_GC_BR/results/20221201192712_base_model/eval_TCGA_stain_normalize_thumbnail/70',
# ]
compare_dirs = [
    '/media/kwaklab_103/sda/data/thumbnail/TCGA_Stomach_452',
    '/media/kwaklab_103/sda/data/thumbnail/TCGA_Stomach_452_stain_normalize_v1_masked',
    '/media/kwaklab_103/sda/data/thumbnail/TCGA_Stomach_452_stain_normalize_thumbnail_KBSMC'
]

MAX_SIZE = 2048

if __name__ == '__main__':
    data_dict = {}

    for current_dir in compare_dirs:
        for file_path in os_walk(current_dir, 'images'):
            filename = os.path.basename(file_path)
            if filename not in data_dict:
                data_dict[filename] = {}
            data_dict[filename][current_dir] = file_path
    print(len(data_dict))

    for filename, dir_data in data_dict.items():
        if len(dir_data.keys()) != len(compare_dirs):
            continue

        print(filename)

        images = []
        for current_dir, file_path in data_dict[filename].items():
            image = cv2.imread(file_path)
            if len(images) > 0:
                image = cv2.resize(image, (images[0].shape[1], images[0].shape[0]))
            images.append(image)

        output = np.hstack(images)
        ratio = MAX_SIZE / max(output.shape[:2])
        output = cv2.resize(output, (int(output.shape[1] * ratio), int(output.shape[0] * ratio)))
        cv2.imshow('T', output)
        cv2.waitKey(0)

