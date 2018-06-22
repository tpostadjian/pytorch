import numbers
import random
import torch
import numpy as np


class CenterCrop:

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        cls_name, cls, img = sample['class_name'], sample['class_code'], sample['image']
        img = sample['image']

        h, w = img.shape[1:]
        th, tw = self.size
        if w == tw and h == th:
            return sample

        x1 = int(round(w - tw) / 2.)
        y1 = int(round(h - th) / 2.)
        img = img[:, x1:x1+tw, y1:y1+th]

        sample = {'class_name': cls_name, 'class_code': cls, 'image': img}

        return sample


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        cls_name, cls, img = sample['class_name'], sample['class_code'], sample['image']
        if random.random() < self.p:
            return sample
        else:
            img = img[:, ::-1, :]
            sample = {'class_name': cls_name, 'class_code': cls, 'image': img}
            return sample


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        cls_name, cls, img = sample['class_name'], sample['class_code'], sample['image']
        if random.random() < self.p:
            return sample
        else:
            img = img[:, :, ::-1]
            sample = {'class_name': cls_name, 'class_code': cls, 'image': img}
            return sample


class ToTorchTensor:
    def __call__(self, sample):
        cls_name, cls, img = sample['class_name'], sample['class_code'], sample['image']
        print(img.shape)
        sample = {'class_name': cls_name, 'class_code': cls, 'image': torch.from_numpy(np.flip(img, axis=0).copy())}
        return sample
