from torch.utils.data import Dataset, DataLoader
import torch
import random
import numpy as np
import gdal
import os


class SPOT_dataset(Dataset):

    def __init__(self, patch_ids, data_img, label_img, window_size, cache=False):
        super(SPOT_dataset, self).__init__()

        self.data_files = [data_img.format(id) for id in patch_ids]
        self.label_files = [label_img.format(id) for id in patch_ids]
        self.window_shape = window_size
        self.cache = cache

        for f in self.data_files + self.label_files:
            if not os.path.isfile(f):
                raise KeyError('{} : not a file '.format(f))

        # Initialize cache dicts
        # if self.cache:
        self.data_cache_ = {}
        self.label_cache_ = {}

    def __len__(self):
        return 10000

    def __getitem__(self, item):
        random_idx = random.randint(0, len(self.data_files) - 1)

        # If the tile hasn't been loaded yet, put in cache
        if random_idx in self.data_cache_.keys():
            data = self.data_cache_[random_idx]
        else:
            # Data is normalized in [0, 1]
            data = gdal.Open(self.data_files[random_idx])
            data = data.ReadAsArray()
            data = 1. / 255 * np.asarray(data, dtype='float32')
            if self.cache:
                self.data_cache_[random_idx] = data

        if random_idx in self.label_cache_.keys():
            label = self.label_cache_[random_idx]
        else:
            # Labels are converted from RGB to their numeric values
            label = gdal.Open(self.label_files[random_idx])
            label = label.ReadAsArray()
            label = np.asarray(label, dtype='int16')
            if self.cache:
                self.label_cache_[random_idx] = label

        def get_random_pos(img, window_shape):
            """ Extract of 2D random patch of shape window_shape in the image """
            w, h = window_shape
            W, H = img.shape[-2:]
            x1 = random.randint(0, W - w - 1)
            x2 = x1 + w
            y1 = random.randint(0, H - h - 1)
            y2 = y1 + h
            return x1, x2, y1, y2

        # Get a random patch while ensuring it contains actual data / groundtruth
        # x1, x2, y1, y2 = get_random_pos(data, self.window_shape)
        # data_p = data[:, x1:x2, y1:y2]
        # label_p = label[x1:x2, y1:y2]
        # while np.all(np.equal(data[0, x1:x2, y1:y2], np.zeros(self.window_shape))) or np.all(np.equal(label[x1:x2, y1:y2], np.zeros(self.window_shape))):
        x1, x2, y1, y2 = get_random_pos(data, self.window_shape)
        data_p = data[:, x1:x2, y1:y2]
        label_p = label[x1:x2, y1:y2]

        data_p, label_p = self.data_augmentation(True, True, data_p, label_p)
        # Return the torch.Tensor values
        return (torch.from_numpy(np.flip(data_p, axis=0).copy()),
                torch.from_numpy(np.flip(label_p, axis=0).copy()))

    @classmethod
    def data_augmentation(cls, v_flip, h_flip, *arrays):
        """
        arrays : zipped data to tranform
        v_flip, h_flip : vertical & horizontal flip flags
        """
        will_v_flip, will_h_flip = False, False
        if v_flip and random.random() < 0.5:
            will_v_flip = True
        if h_flip and random.random() < 0.5:
            will_h_flip = True

        flip = []
        for a in arrays:
            if will_v_flip:
                if len(a.shape) == 2:
                    a = a[::-1, :]
                else:
                    a = a[:, ::-1, :]
            if will_h_flip:
                if len(a.shape) == 2:
                    a = a[:, ::-1]
                else:
                    a = a[:, :, ::-1]
            flip.append(a)

        return tuple(flip)