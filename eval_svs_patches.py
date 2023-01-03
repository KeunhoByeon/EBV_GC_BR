import argparse
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from dataloader import EBVGCDataset
from model import Classifier
from tools.utils import os_walk, load_annotation, resize_and_pad_image, get_thumbnail


def prepare_eval_dataset(patch_dir, svs_dir, mask_dir, anno_path, label_if_not_exist=None):
    data = {}

    anno_data = load_annotation(anno_path)
    for file_index, label in anno_data.items():
        label = 'Negative' if label == 0 else label
        label = 'Positive' if label == 1 else label
        anno_data[file_index] = label

    for svs_path in os_walk(svs_dir, '.svs'):
        filename = os.path.basename(svs_path)
        file_full_index = filename.replace(os.path.splitext(filename)[-1], '').lower()
        file_index = '-'.join(file_full_index.split('-')[:3])
        if file_full_index not in data.keys():
            label = anno_data[file_index.upper()] if file_index.upper() in anno_data.keys() else label_if_not_exist
            data[file_full_index] = {'patches': [], 'svs_path': None, 'mask_path': None, 'label': label}
        data[file_full_index]['svs_path'] = svs_path

    for patch_path in os_walk(patch_dir, ('.png', '.jpg', '.jpeg')):
        file_full_index = os.path.basename(patch_path).split('_patch')[0].lower()
        file_index = '-'.join(file_full_index.split('-')[:3])
        if file_full_index not in data.keys():
            label = anno_data[file_index.upper()] if file_index.upper() in anno_data.keys() else label_if_not_exist
            data[file_full_index] = {'patches': [], 'svs_path': None, 'mask_path': None, 'label': label}
        data[file_full_index]['patches'].append(patch_path)

    for mask_path in os_walk(mask_dir, ('.png', '.jpg', '.jpeg')):
        filename = os.path.basename(mask_path)
        file_full_index = filename.replace(os.path.splitext(filename)[-1], '').lower()
        file_index = '-'.join(file_full_index.split('-')[:3])
        if file_full_index not in data.keys():
            label = anno_data[file_index.upper()] if file_index.upper() in anno_data.keys() else label_if_not_exist
            data[file_full_index] = {'patches': [], 'svs_path': None, 'mask_path': None, 'label': label}
        data[file_full_index]['mask_path'] = mask_path

    return data


def check_data(data):
    cnt, failed, cnt_dict = 0, [0, 0, 0, 0], {}
    indices_to_del = []
    for i, (file_index, data_dict) in enumerate(data.items()):
        patches = data_dict['patches']
        svs_path = data_dict['svs_path']
        mask_path = data_dict['mask_path']
        label = data_dict['label']
        if len(patches) == 0:
            failed[0] += 1
            indices_to_del.append(file_index)
            continue
        elif svs_path is None:
            failed[1] += 1
            indices_to_del.append(file_index)
            continue
        elif mask_path is None:
            failed[2] += 1
            indices_to_del.append(file_index)
            continue
        elif label is None:
            failed[3] += 1
            indices_to_del.append(file_index)
            continue
        if label not in cnt_dict.keys():
            cnt_dict[label] = 0
        cnt_dict[label] += 1
        cnt += 1
    for file_index in indices_to_del:
        del data[file_index]
    print("Total Cases: {} ({}), Failed: {}".format(len(data), cnt_dict, failed))

    return data


def evaluate(model, eval_loader, desc=''):
    model.eval()

    outputs = {}
    with torch.no_grad():
        for i, (img_paths, inputs, _) in tqdm(enumerate(eval_loader), leave=False, desc='Evaluating {}'.format(desc), total=len(eval_loader)):
            if torch.cuda.is_available():
                inputs = inputs.cuda()

            output = model(inputs)
            preds = torch.argmax(output, dim=1)
            for img_path, pred in zip(img_paths, preds):
                # pred = 'Benign' if pred == 0 else pred
                # pred = 'Positive' if pred == 1 else pred
                # pred = 'Negative' if pred == 2 else pred
                outputs[img_path] = pred.item()

    return outputs


def run(args):
    # Model
    model = Classifier(args.model, num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.checkpoint))

    # CUDA
    if torch.cuda.is_available():
        model = model.cuda()

    data = prepare_eval_dataset(args.patch_dir, args.svs_dir, args.mask_dir, args.anno_path, label_if_not_exist=args.label_if_not_exist)
    data = check_data(data)

    with open(os.path.join(args.result, 'results.csv'), 'w') as wf:
        wf.write('file,x,y,pred,target\n')

    for i, (file_index, data_dict) in enumerate(data.items()):
        patches = data_dict['patches']
        svs_path = data_dict['svs_path']
        mask_path = data_dict['mask_path']
        label = data_dict['label']

        if len(patches) == 0:
            continue
        elif svs_path is None:
            continue

        # Dataset
        samples = list(zip(patches, np.zeros(len(patches))))
        eval_dataset = EBVGCDataset(samples, input_size=args.input_size, is_train=False, stain_norm_path=args.stain_target)
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)

        # Run Evaluating
        outputs = evaluate(model, eval_loader, desc='{}/{}'.format(i, len(data)))

        # Load Thumbnail
        thumbnail, num_pixels, thumbnail_ratio = get_thumbnail(svs_path, return_info=True)

        # Load Mask
        if mask_path is not None:
            tissue_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            tissue_mask = resize_and_pad_image(tissue_mask, target_size=thumbnail.shape, padding=False)
            tissue_mask = cv2.cvtColor(tissue_mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            tissue_mask = tissue_mask.astype(float)
            tissue_mask /= 255.
        else:
            tissue_mask = np.ones_like(thumbnail)
            tissue_mask = tissue_mask.astype(float)

        # Load Label
        label = 'None' if label is None else label

        # Make Result Image
        results_data = []
        output_mask = np.zeros(thumbnail.shape)
        for img_path, pred in outputs.items():
            coord_x = int(os.path.basename(img_path).split('.')[1].split('_')[2])
            coord_y = int(os.path.basename(img_path).split('.')[1].split('_')[3])

            coord_x_1 = coord_x * thumbnail_ratio
            coord_y_1 = coord_y * thumbnail_ratio
            coord_x_2 = coord_x_1 + args.patch_size * thumbnail_ratio
            coord_y_2 = coord_y_1 + args.patch_size * thumbnail_ratio

            coord_x_1, coord_y_1 = int(coord_x_1), int(coord_y_1)
            coord_x_2, coord_y_2 = int(coord_x_2), int(coord_y_2)

            if pred == 0:
                output_mask[coord_y_1:coord_y_2, coord_x_1:coord_x_2, 0] = 0
                output_mask[coord_y_1:coord_y_2, coord_x_1:coord_x_2, 1] = 0
                output_mask[coord_y_1:coord_y_2, coord_x_1:coord_x_2, 2] = 0
            elif pred == 1:
                output_mask[coord_y_1:coord_y_2, coord_x_1:coord_x_2, 0] = 0
                output_mask[coord_y_1:coord_y_2, coord_x_1:coord_x_2, 1] = 1
                output_mask[coord_y_1:coord_y_2, coord_x_1:coord_x_2, 2] = 0
            elif pred == 2:
                output_mask[coord_y_1:coord_y_2, coord_x_1:coord_x_2, 0] = 1
                output_mask[coord_y_1:coord_y_2, coord_x_1:coord_x_2, 1] = 0
                output_mask[coord_y_1:coord_y_2, coord_x_1:coord_x_2, 2] = 0

            results_data.append('{},{},{},{},{}\n'.format(os.path.basename(img_path), coord_x, coord_y, pred, label))

        with open(os.path.join(args.result, 'results.csv'), 'a') as wf:
            wf.writelines(results_data)

        result_mask = (output_mask * 2 + 1) * tissue_mask
        output = (result_mask + 1) / 3 * thumbnail
        output = np.clip(output, 0, 255).astype(np.uint8)

        save_path = os.path.join(args.result, os.path.basename(svs_path).replace('.svs', '_class_{}.png'.format(label)))
        cv2.imwrite(save_path, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training')
    # Model Arguments
    parser.add_argument('--model', default='efficientnet_b0')
    parser.add_argument('--num_classes', default=3, type=int, help='number of classes')
    parser.add_argument('--checkpoint', default=None, type=str, help='path to checkpoint')
    parser.add_argument('--checkpoint_name', default='20221129163053_color_noise', type=str)
    parser.add_argument('--checkpoint_epoch', default=83, type=int)
    # Data Paths
    parser.add_argument('--patch_dir', default='/media/kwaklab_103/sdc/data/patch_data/TCGA_Stomach_452_normalized_1', help='path to dataset')
    parser.add_argument('--svs_dir', default='/media/kwaklab_103/sda/data/raw_data/TCGA_Stomach_452', help='path to svs dataset')
    parser.add_argument('--mask_dir', default='/media/kwaklab_103/sda/data/raw_data/TCGA_Stomach_452_tissue_mask', help='path to mask dataset')
    parser.add_argument('--stain_target', default=None)
    # Data Arguments
    parser.add_argument('--anno_path', default='/media/kwaklab_103/sda/data/raw_data/TCGA_Stomach_452/STAD_molecular_subtype TCGA data.xlsx', help='path to svs dataset')
    parser.add_argument('--input_size', default=512, type=int, help='image input size')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
    parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size')
    # Debugging Arguments
    parser.add_argument('--patch_size', default=1024, type=int, help='num pixels of patch')
    parser.add_argument('--label_if_not_exist', default=None)
    parser.add_argument('--result', default=None, help='path to results')
    parser.add_argument('--result_tag', default='eval_TCGA_Stomach_452_normalized_1')
    args = parser.parse_args()

    # Paths setting
    if args.checkpoint is None or len(args.checkpoint) == 0:
        if args.checkpoint_name is not None and args.checkpoint_epoch is not None:
            args.checkpoint = './results/{}/checkpoints/{}.pth'.format(args.checkpoint_name, args.checkpoint_epoch)
        if args.checkpoint is None or not os.path.isfile(args.checkpoint):
            print('Cannot find checkpoint file!: {} {} {}'.format(args.checkpoint, args.checkpoint_name, args.checkpoint_epoch))
            raise AssertionError

    if args.result is None:
        if args.checkpoint_name is not None and args.checkpoint_epoch is not None:
            args.result = './results/{}/{}/{}'.format(args.checkpoint_name, args.result_tag, args.checkpoint_epoch)
        else:
            print('Please specify result dir: {} {} {} {}'.format(args.result, args.checkpoint_name, args.result_tag, args.checkpoint_epoch))
            raise AssertionError

    args.result = os.path.expanduser(args.result)
    os.makedirs(args.result, exist_ok=True)

    run(args)
