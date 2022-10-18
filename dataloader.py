import cv2
import imgaug.augmenters as iaa
import numpy as np
import torch
from torch.utils.data import Dataset

from utils import resize_and_pad_image


class EBVGCDataset(Dataset):
    def __init__(self, samples, input_size: int = None, do_aug: bool = False):
        self.input_size = input_size
        self.do_aug = do_aug

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        # Augmentation setting (Not yet implemented all)
        self.affine_seq = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Sometimes(0.5, iaa.Affine(rotate=(-45, 45)))
        ], random_order=True)
        # self.color_seq = iaa.Sequential([
        #     iaa.Sometimes(0.1, iaa.GaussianBlur(sigma=(0, 0.5))),
        #     iaa.Sometimes(0.1, iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5)),
        #     iaa.Sometimes(0.2, iaa.Multiply((0.8, 1.2), per_channel=0.2)),
        #     iaa.Sometimes(0.2, iaa.MultiplyHue((0.8, 1.2))),
        #     iaa.Sometimes(0.2, iaa.MultiplySaturation((0.8, 1.2))),
        #     iaa.Sometimes(0.2, iaa.LogContrast((0.8, 1.2))),
        # ], random_order=True)

        self.samples = samples

    def __getitem__(self, index):
        img_path, gt = self.samples[index]

        img = cv2.imread(img_path)
        if self.input_size is not None:
            img = resize_and_pad_image(img, target_size=(self.input_size, self.input_size), keep_ratio=True, padding=True)

        if self.do_aug:
            img = self.affine_seq.augment_image(img)
            # Not yet implemented
            # img = self.color_seq.augment_image(img)

        img = img.astype(np.float32) / 255.
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        gt = torch.from_numpy(np.array(gt)).type(torch.LongTensor)

        return img_path, img, gt

    def __len__(self):
        return len(self.samples)
