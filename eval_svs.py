import os.path

import cv2
import numpy as np
import openslide
import torch
from tqdm import tqdm

from model import Classifier
from tools.utils import resize_and_pad_image, load_annotation


def preprocess(img):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    img = resize_and_pad_image(img, target_size=(512, 512), keep_ratio=True, padding=True)

    img = img.astype(np.float32) / 255.
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(np.array([img]))

    return img


def eval_svs(model, svs_path, anno_data, save_dir, read_size=1024, thumbnail_size=1024, mean_thresh=240):
    try:
        file_index = '-'.join(filename.split('.')[0].split('-')[:3])
        if file_index not in anno_data.keys():  # TEMP
            return
        label = anno_data[file_index]
        label = 2 if label == 0 else label

        if label != 1:  # TEMP
            return

        slide = openslide.OpenSlide(svs_path)

        w_pixels, h_pixels = slide.level_dimensions[0]
        num_w, num_h = w_pixels // read_size, h_pixels // read_size
        ratio = min(thumbnail_size // num_w, thumbnail_size // num_h)
        thumbnail_shape = (num_w * ratio, num_h * ratio)
        thumbnail = slide.get_thumbnail(thumbnail_shape)
        thumbnail = np.array(thumbnail)
        thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2BGR)

        pred_num = [0, 0, 0]
        thumbnail_seg = thumbnail.copy().astype(float)
        with torch.no_grad():
            progress = tqdm(range(num_h * num_w))
            for i in progress:
                h_i = i // num_w
                w_i = i % num_w

                slide_img = slide.read_region((read_size * w_i, read_size * h_i), 0, (read_size, read_size))
                slide_img = np.array(slide_img)

                if np.mean(slide_img) > mean_thresh:
                    pred = -1
                else:
                    slide_img = cv2.cvtColor(slide_img, cv2.COLOR_RGB2BGR)
                    input_img = preprocess(slide_img.copy())

                    if torch.cuda.is_available():
                        input_img = input_img.cuda()  # CUDA

                    output = model(input_img)
                    _, preds = output.topk(1, 1, True, True)
                    pred = preds[0].item()

                if pred == -1:
                    thumbnail_seg[h_i * ratio: (h_i + 1) * ratio, w_i * ratio: (w_i + 1) * ratio, 0] /= 4
                    thumbnail_seg[h_i * ratio: (h_i + 1) * ratio, w_i * ratio: (w_i + 1) * ratio, 1] /= 4
                    thumbnail_seg[h_i * ratio: (h_i + 1) * ratio, w_i * ratio: (w_i + 1) * ratio, 2] /= 4
                elif pred == 0:
                    thumbnail_seg[h_i * ratio: (h_i + 1) * ratio, w_i * ratio: (w_i + 1) * ratio, 0] /= 2
                    thumbnail_seg[h_i * ratio: (h_i + 1) * ratio, w_i * ratio: (w_i + 1) * ratio, 1] /= 2
                    thumbnail_seg[h_i * ratio: (h_i + 1) * ratio, w_i * ratio: (w_i + 1) * ratio, 2] /= 2
                    pass
                elif pred == 1:
                    thumbnail_seg[h_i * ratio: (h_i + 1) * ratio, w_i * ratio: (w_i + 1) * ratio, 0] /= 2
                    thumbnail_seg[h_i * ratio: (h_i + 1) * ratio, w_i * ratio: (w_i + 1) * ratio, 1] += 127
                    thumbnail_seg[h_i * ratio: (h_i + 1) * ratio, w_i * ratio: (w_i + 1) * ratio, 2] /= 2
                elif pred == 2:
                    thumbnail_seg[h_i * ratio: (h_i + 1) * ratio, w_i * ratio: (w_i + 1) * ratio, 0] += 127
                    thumbnail_seg[h_i * ratio: (h_i + 1) * ratio, w_i * ratio: (w_i + 1) * ratio, 1] /= 2
                    thumbnail_seg[h_i * ratio: (h_i + 1) * ratio, w_i * ratio: (w_i + 1) * ratio, 2] /= 2

                pred_num[pred] += 1
                progress.set_description(desc='{} {} {}'.format(file_index, label, pred_num))

                sample_path = os.path.join(save_dir, '{}_class_{}_sample.png'.format(file_index, label))
                if num_h * num_w * 45 // 100 < i < num_h * num_w * 55 // 100 and pred == label and not os.path.isfile(sample_path):
                    cv2.imwrite(sample_path, slide_img)

        thumbnail_seg = np.clip(thumbnail_seg, 0, 255)
        cv2.imwrite(os.path.join(save_dir, '{}_class_{}_thumbnail.png'.format(file_index, label)), thumbnail)
        cv2.imwrite(os.path.join(save_dir, '{}_class_{}_output.png'.format(file_index, label)), thumbnail_seg.astype(np.uint8))
    except:
        print('Error: {}'.format(svs_path))
        print()


if __name__ == '__main__':
    svs_dir = '/media/kwaklab_103/sda/data/raw_data/TCGA_Stomach_452/'
    xlsx_path = '/media/kwaklab_103/sda/data/raw_data/TCGA_Stomach_452/STAD_molecular_subtype TCGA data.xlsx'
    model_path = './results/20221104062515_lr4e_06/checkpoints/93.pth'
    save_dir = './debug/20221104062515_lr4e_06_epoch93_new_20221108'

    anno_data = load_annotation(xlsx_path)

    model = Classifier('efficientnet_b0', num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()  # CUDA

    os.makedirs(save_dir, exist_ok=True)
    for path, dir, files in os.walk(svs_dir):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext != '.svs':
                continue
            svs_path = os.path.join(path, filename)
            eval_svs(model, svs_path, anno_data, save_dir)
