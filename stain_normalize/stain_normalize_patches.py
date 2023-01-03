import os

import cv2
from joblib import Parallel, delayed
from tiatoolbox.tools.stainnorm import MacenkoNormalizer
from tqdm import tqdm

from tools.utils import os_walk

target_version = 1
n_jobs = 12

target_path = "./data/target_{}.png".format(target_version)
input_dir = "/media/kwaklab_103/sdb/data/patch_data/TCGA_Stomach_452"
output_dir = "/media/kwaklab_103/sdc/data/patch_data/TCGA_Stomach_452_normalized_{}".format(target_version)


def make_stain_normalized(source_path, normalizer):
    save_path = source_path.replace(input_dir, output_dir)

    source = cv2.imread(source_path, cv2.IMREAD_COLOR)
    source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)

    try:
        transformed = normalizer.transform(source)
        transformed = cv2.cvtColor(transformed, cv2.COLOR_RGB2BGR)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, transformed)
    except Exception as e:
        print(e)
        source = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)
        os.makedirs('./data/failed/{}'.format(os.path.basename(output_dir)), exist_ok=True)
        cv2.imwrite('./data/failed/{}/{}'.format(os.path.basename(output_dir), os.path.basename(source_path)), source)
        return


if __name__ == "__main__":
    target = cv2.imread(target_path)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

    normalizer = MacenkoNormalizer()
    normalizer.fit(target)

    source_paths = []
    for source_path in os_walk(input_dir, ext=('.png', '.jpg', '.jpeg')):
        source_paths.append(source_path)

    if n_jobs > 1:
        Parallel(n_jobs=n_jobs)(delayed(make_stain_normalized)(source_path, normalizer) for source_path in tqdm(source_paths))
    else:
        for source_path in tqdm(source_paths):
            make_stain_normalized(source_path, normalizer)
