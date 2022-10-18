import os
import cv2
import numpy as np
import random


def make_tile_images(txt_path, save_dir, num_img=256, num_line=16, img_size=32):
    img_paths = [[[] for _ in range(3)] for _ in range(3)]
    with open(txt_path, 'r') as rf:
        arguments = rf.readline()
        for line in rf.readlines():
            if '*Evaluation' in line:
                break
            line_split = line.replace('\n', '').split(',')
            img_path, gt, pred = line_split[0], int(line_split[1]), int(line_split[2])
            img_paths[gt][pred].append(img_path)

    total_output, total_output_line = [], []
    for gt_i in range(3):
        for pred_i in range(3):
            random.shuffle(img_paths[gt_i][pred_i])
            print(gt_i, pred_i, len(img_paths[gt_i][pred_i]))
            output_img, output_img_line = [], []
            for i in range(num_img):
                if i >= len(img_paths[gt_i][pred_i]):
                    i = i % len(img_paths[gt_i][pred_i])

                img = cv2.imread(img_paths[gt_i][pred_i][i])
                img = cv2.resize(img, (img_size, img_size))

                output_img_line.append(img)
                if len(output_img_line) == num_line:
                    output_img.append(np.hstack(output_img_line))
                    output_img_line = []

            output_img = np.vstack(output_img)
            cv2.imwrite(os.path.join(save_dir, '{}_{}.png'.format(gt_i, pred_i)), output_img)
            total_output_line.append(output_img)

        total_output.append(np.hstack(total_output_line))
        total_output_line = []

    total_output = np.vstack(total_output)
    cv2.imwrite(os.path.join(save_dir, 'total.png'), total_output)


if __name__ == '__main__':
    input_dir = '../results/20221018161257/eval_valset/50'
    log_path = os.path.join(input_dir, 'eval_log.txt')
    save_dir = os.path.join(input_dir, 'debug')
    os.makedirs(save_dir, exist_ok=True)
    random.seed(103)

    make_tile_images(log_path, save_dir)
