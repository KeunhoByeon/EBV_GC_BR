import multiprocessing as mp
import os

import cv2
import numpy as np
from joblib import Parallel, delayed
from tiatoolbox.tools.stainnorm import VahadaneNormalizer
from tqdm import tqdm

from utils import os_walk

target_path = "/media/kwaklab_103/sda/data/patch_data/KBSMC/gastric/gastric_EBV_1024/Gastric_WSI/gastric_wsi_1024_08/S 2012017007/patch_1343_class_2.jpg"
input_dir = "/media/kwaklab_103/sdb/data/patch_data/TCGA_Stomach_452"
output_dir = "/media/kwaklab_103/sdb/data/patch_data/TCGA_Stomach_452_stain_normalize"


def make_stain_normalized(source_path, normalizer_vahadane):
    save_path = source_path.replace(input_dir, output_dir)

    source = cv2.imread(source_path, cv2.IMREAD_COLOR)
    source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    if np.sum(source[:, -10:]) <= 0:
        source = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, source)
        return
    elif np.mean(source[:, :, 0]) > 230 and np.mean(source[:, :, 1]) > 230 and np.mean(source[:, :, 2]) > 230:
        source = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, source)
        return

    try:
        transformed_vahadane = normalizer_vahadane.transform(source)
        transformed_vahadane = cv2.cvtColor(transformed_vahadane, cv2.COLOR_RGB2BGR)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, transformed_vahadane)
    except Exception as e:
        print(e)
        os.makedirs(os.path.expanduser('~/data/failed'), exist_ok=True)
        source = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.expanduser('~/data/failed/{}'.format(os.path.basename(source_path))), source)
        return


if __name__ == "__main__":
    target = cv2.imread(target_path)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

    normalizer_vahadane = VahadaneNormalizer()
    normalizer_vahadane.fit(target)

    source_paths = []
    for source_path in os_walk(input_dir, ext=('.png', '.jpg', '.jpeg')):
        save_path = source_path.replace(input_dir, output_dir)
        if os.path.isfile(save_path):
            continue
        source_paths.append(source_path)

    n_jopbs = mp.cpu_count() - 1
    Parallel(n_jobs=n_jopbs)(delayed(make_stain_normalized)(source_path, normalizer_vahadane) for source_path in tqdm(source_paths))
