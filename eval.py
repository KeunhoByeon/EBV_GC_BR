import argparse
import os
import time

import torch
import torch.nn as nn
from tqdm import tqdm

from data_utils import prepare_gastric_EBV_data_json
from dataloader import EBVGCDataset
from logger import Logger
from model import Classifier
from utils import accuracy


def eval(model, criterion, eval_loader, logger=None):
    model.eval()

    with torch.no_grad():
        confusion_mat = [[0 for _ in range(3)] for _ in range(3)]
        for i, (img_paths, inputs, targets) in tqdm(enumerate(eval_loader), leave=False, desc='Evaluating', total=len(eval_loader)):
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()

            output = model(inputs)
            loss = criterion(output, targets)
            acc = accuracy(output, targets)[0]

            # Confusion Matrix
            _, preds = output.topk(1, 1, True, True)
            for img_path, t, p in zip(img_paths, targets, preds):
                confusion_mat[int(t.item())][p[0].item()] += 1
                logger.write_log('{},{},{}'.format(img_path, int(t.item()), p[0].item()))  # img path, GT, pred

            logger.add_history('total', {'loss': loss.item(), 'accuracy': acc})

            del output, loss, acc

        if logger is not None:
            logger('*Evaluation', history_key='total', confusion_mat=confusion_mat, time=time.strftime('%Y%m%d%H%M%S'))


def run(args):
    # Model
    model = Classifier(args.model, num_classes=args.num_classes, pretrained=True)
    model.load_state_dict(torch.load(args.checkpoint))

    # Criterion
    criterion = nn.CrossEntropyLoss()

    # CUDA
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    # Dataset
    if args.dataset_type == 'val':
        _, eval_set, _ = prepare_gastric_EBV_data_json(args.data)
    elif args.dataset_type == 'test':
        _, _, eval_set = prepare_gastric_EBV_data_json(args.data)
    else:
        print('Dataset type {} is not valid'.format(args.dataset_type))
        raise AssertionError
    eval_dataset = EBVGCDataset(eval_set, input_size=args.input_size, do_aug=False)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=False)

    # Logger
    logger = Logger(os.path.join(args.result, 'eval_log.txt'), epochs=1, dataset_size=len(eval_loader.dataset), float_round=5)
    logger.set_sort(['loss', 'accuracy', 'confusion_mat', 'time'])
    logger(str(args))

    # Run training
    eval(model, criterion, eval_loader, logger=logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training')
    # Model Arguments
    parser.add_argument('--model', default='efficientnet_b0')
    parser.add_argument('--num_classes', default=3, type=int, help='number of classes')
    parser.add_argument('--checkpoint', default=None, type=str, help='path to checkpoint')
    parser.add_argument('--checkpoint_name', default='20221016021717_b0', type=str)
    parser.add_argument('--checkpoint_epoch', default=58, type=int)
    # Data Arguments
    parser.add_argument('--data', default='/media/kwaklab_103/sda/data/patch_data/KBSMC/gastric/gastric_EBV_1024', help='path to dataset')
    parser.add_argument('--input_size', default=512, type=int, help='image input size')
    parser.add_argument('--workers', default=8, type=int, help='number of data loading workers')
    parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size')
    parser.add_argument('--dataset_type', default='val', type=str, choices=['val', 'test'], help='val or test')
    # Debugging Arguments
    parser.add_argument('--result', default=None, help='path to results')
    args = parser.parse_args()

    # Paths setting
    if args.checkpoint is None:
        if args.checkpoint_name is not None and args.checkpoint_epoch is not None:
            args.checkpoint = './results/{}/models/{}.pth'.format(args.checkpoint_name, args.checkpoint_epoch)
        if args.checkpoint is None or not os.path.isfile(args.checkpoint):
            print('Cannot find checkpoint file!: {} {} {}'.format(args.checkpoint, args.checkpoint_name, args.checkpoint_epoch))
            raise AssertionError

    if args.result is None:
        if args.checkpoint_name is not None and args.checkpoint_epoch is not None:
            args.result = './results/{}/eval_{}set/{}'.format(args.checkpoint_name, args.dataset_type, args.checkpoint_epoch)
        else:
            print('Please specify result dir: {} {} {} {}'.format(args.result, args.dataset_type, args.checkpoint_name, args.checkpoint_epoch))
            raise AssertionError

    args.data = os.path.expanduser(args.data)
    args.result = os.path.expanduser(args.result)
    os.makedirs(args.result, exist_ok=True)

    run(args)
