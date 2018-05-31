import torch.utils.data as data
import gdal
import numpy as np


class SPOTDataset_test(data.Dataset):

    def __init__(self, img, patch_size):

        self.img = img
        self.data = self.read_img()
        self.offset = int(patch_size / 2)

    def __len__(self):
        return self.data.shape[0]*self.data.shape[1]

    def __getitem__(self, item):
        i = item[0]
        j = item[1]
        return self.data[:, i - self.offset:i + self.offset + 1, j - self.offset:j + self.offset + 1]

    def read_img(self):
        ds = gdal.Open(self.img)
        ds = ds.ReadAsArray()
        ds = 1. / 255 * np.asarray(ds, dtype='float32')
        return ds
