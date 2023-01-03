import os

import cv2
import numpy as np
import openslide
from tqdm import tqdm

from tools.utils import load_annotation

FILE_INDEX = 'TCGA-BR-6705'
# FILE_INDEX = None
STEP_SIZES = [0.1, 0.3, 0.5, 1.0, 3.0, 5.0, 10.0, 20.0]
INITIAL_STEP_SIZE_INDEX = 3

svs_dir = '/media/kwaklab_103/sda/data/raw_data/TCGA_Stomach_452/'
xlsx_path = '/media/kwaklab_103/sda/data/raw_data/TCGA_Stomach_452/STAD_molecular_subtype TCGA data.xlsx'


def get_svs_path(svs_dir, file_index):
    for path, dir, files in os.walk(svs_dir):
        for filename in files:
            if os.path.splitext(filename)[-1] != '.svs':
                continue
            if '-'.join(filename.split('.')[0].split('-')[:3]) == file_index:
                return os.path.join(path, filename)


def get_svs_paths(svs_dir, anno_data=None):
    svs_paths = {}
    for path, dir, files in os.walk(svs_dir):
        for filename in files:
            if os.path.splitext(filename)[-1] != '.svs':
                continue
            file_index = '-'.join(filename.split('.')[0].split('-')[:3])
            if anno_data is not None and file_index not in anno_data.keys():
                continue
            svs_paths[file_index] = os.path.join(path, filename)
    return svs_paths


def infinite_zeros():
    while True:
        yield 0


def check_svs(svs_path, label, read_size=1024):
    print('Label: {}'.format(label))

    slide = openslide.OpenSlide(svs_path)
    w_pixels, h_pixels = slide.level_dimensions[0]
    print(slide.properties)
    print('Slide Level Dimensions: {}'.format(slide.level_dimensions))
    print('Pixels: {}, {}'.format(w_pixels, h_pixels))

    ratio = min(1024 / w_pixels, 1024 / h_pixels)
    thumbnail_shape = (int(w_pixels * ratio), int(h_pixels * ratio))
    print('Thumbnail Shape: {}'.format(thumbnail_shape))

    thumbnail = slide.get_thumbnail(thumbnail_shape)
    thumbnail = np.array(thumbnail)
    thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2BGR)

    step_index = INITIAL_STEP_SIZE_INDEX

    wi, hi = 0, 0
    progress = tqdm(infinite_zeros(), leave=False)
    for _ in progress:
        thumbnail_temp = thumbnail.copy()
        pt1 = (int(wi / w_pixels * thumbnail_shape[0]), int(hi / h_pixels * thumbnail_shape[1]))
        pt2 = (int((wi + read_size) / w_pixels * thumbnail_shape[0]), int((hi + read_size) / h_pixels * thumbnail_shape[1]))
        thumbnail_temp = cv2.rectangle(thumbnail_temp, pt1, pt2, (127, 255, 0), thickness=2)

        slide_img = slide.read_region((wi, hi), 0, (read_size, read_size))
        slide_img = np.array(slide_img)
        slide_img = cv2.resize(slide_img, (1024, 1024))
        slide_img = cv2.cvtColor(slide_img, cv2.COLOR_RGB2BGR)

        slide_mean = round(np.mean(slide_img), 4)
        slide_std = round(np.std(slide_img), 4)
        slide_b_mean = round(np.mean(slide_img[:, :, 0]), 4)
        slide_g_mean = round(np.mean(slide_img[:, :, 1]), 4)
        slide_r_mean = round(np.mean(slide_img[:, :, 2]), 4)
        slide_channel_std = round(np.std([slide_b_mean, slide_g_mean, slide_r_mean]), 4)
        color_info = '{} {} {}'.format(slide_mean, slide_std, slide_channel_std)

        read_step = int(read_size * STEP_SIZES[step_index])

        cv2.imshow('Slide', slide_img)
        cv2.imshow('Thumbnail', thumbnail_temp)
        # cv2.imshow('slide_img', cv2.cvtColor(slide_img, cv2.COLOR_BGR2GRAY))
        # cv2.imshow('Thumbnail', cv2.cvtColor(thumbnail_temp, cv2.COLOR_BGR2GRAY))

        desc = 'Location {}/{},{}/{}  Slide Pixels {}  Step Size {}  {}'.format(hi, h_pixels, wi, w_pixels, read_size, STEP_SIZES[step_index], color_info)
        progress.set_description(desc=desc)

        key = cv2.waitKey(0)
        if key == ord('w'):  # Move
            hi -= read_step
        elif key == ord('a'):
            wi -= read_step
        elif key == ord('s'):
            hi += read_step
        elif key == ord('d'):
            wi += read_step
        elif key == ord('f'):  # Step size
            step_index -= 1
        elif key == ord('r'):
            step_index += 1
        elif key == ord('g'):
            step_index = 0
        elif key == ord('t'):
            step_index = len(STEP_SIZES) - 1
        elif key == ord('z'):  # Zoom
            read_size -= 128
        elif key == ord('c'):
            read_size += 128
        elif key == ord('c'):
            read_size = 1024
        elif key == ord('q'):  # Quit
            exit(0)
        elif key == ord('n'):  # Next/Prev File
            return 1
        elif key == ord('b'):
            return -1

        read_size = max(128, min(read_size, 4096))
        wi = max(0, min(wi, w_pixels - read_size))
        hi = max(0, min(hi, h_pixels - read_size))
        step_index = max(0, min(step_index, len(STEP_SIZES) - 1))


if __name__ == '__main__':
    anno_data = load_annotation(xlsx_path)

    if FILE_INDEX is not None:
        svs_paths = {FILE_INDEX: get_svs_path(svs_dir, FILE_INDEX)}
    else:
        svs_paths = get_svs_paths(svs_dir, anno_data=anno_data)

    i = 0
    file_indices = sorted(list(svs_paths.keys()))
    while True:
        file_index = file_indices[i]
        svs_path = svs_paths[file_index]
        print(svs_path)
        label = anno_data[file_index] if file_index in anno_data.keys() else 'None'
        label = 'Negative' if label == 0 else label
        label = 'Positive' if label == 1 else label

        print('[{}]'.format(file_index))
        i += check_svs(svs_path, label)
        i = max(0, min(len(svs_paths) - 1, i))
        print()
