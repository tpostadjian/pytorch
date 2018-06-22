import numpy as np
import numbers
import random


class CenterCrop:

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        img = sample['image']
        print(img)
        h, w = img.shape[1:]
        th, tw = self.size
        if w == tw and h == th:
            return sample

        x1 = int(round(w - tw) / 2.)
        y1 = int(round(h - th) / 2.)
        img = img[:, x1:x1+tw, y1:y1+th]
        sample['image'] = img
        return sample


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            img = sample['image']
            img = img[:, ::-1, :]
            sample['image'] = img
            return sample


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            img = sample['image']
            img = img[:, :, ::-1]
            sample['image'] = img
            return sample

