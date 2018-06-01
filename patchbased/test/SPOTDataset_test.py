import torch.utils.data as data
import gdal
import numpy as np
import torch


class SPOTDataset_test(data.Dataset):
    def __init__(self, img, patch_size):
        self.img = img
        self.patch_size = patch_size
        self.offset = int(self.patch_size / 2)
        self.data = self.read_img()
        self.data_noedge = self.data[:,
                           self.offset:self.data.shape[1] - self.offset,
                           self.offset:self.data.shape[2] - self.offset]

    def __len__(self):
        return (self.data.shape[1] - 2 * self.offset) * (self.data.shape[2] - 2 * self.offset)

    def __getitem__(self, item):
        _, nl, nc = self.data_noedge.shape
        l = np.floor_divide(item, nc) + self.offset
        c = np.mod(item, nc) + self.offset
        patch = self.data[:, l - self.offset:l + self.offset + 1, c - self.offset:c + self.offset + 1]
        return patch

    def read_img(self):
        ds = gdal.Open(self.img)
        ds = ds.ReadAsArray()
        ds = 1. / 255 * np.asarray(ds, dtype='float32')
        ds = torch.from_numpy(ds)
        ds = ds.cuda()
        return ds
