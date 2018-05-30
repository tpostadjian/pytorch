import torch
import torch.utils.data as data
import numpy as np
# import h5py
import gdal
import os
import glob as glob
import random


def hdf5_reader(h5_file):
    h5_ds = h5py.File(h5_file, 'r')
    img_list = []
    img_count = 1
    eof = False
    while not eof:
        img_str = 'img_' + str(img_count)
        try:
            img = h5_ds[img_str]
            img = np.array(img)
            img = torch.from_numpy(img)
            img = img.float()
            img_list.append(img)
            img_count += 1
        except KeyError:
            eof = True
            img_count -= 1
            pass
    return img_list, img_count


def tif_reader(rep, ids):
    img_list = []
    img_count = 0
    # get all images in current class training directory
    tif_list = glob.glob(rep + '/*.tif')
    tif_list = [os.path.basename(tif) for tif in tif_list]
    # get only train (valid) samples with basename comparison
    tif_list = list(set(tif_list) & set(ids))
    # re-join path + train (valid) samples name for further processing
    tif_list = [os.path.join(rep, tif) for tif in tif_list]
    for tif in tif_list:
            img = gdal.Open(tif)
            img = img.ReadAsArray()
            img = 1. / 255 * np.asarray(img, dtype='float32')
            img_list.append(img)
            img_count += 1
    return img_list, img_count


class ImageDataset(data.Dataset):
    # ~ ------
    def __init__(self, ids, rootPath, reader='hdf5_reader', transform=True):
        """
        Args:
            ids : ids of training or valid samples
            rootPath (string): Path to the image dataset
            reader : Input file reader - hdf5_reader or tif_reader
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.ids = ids
        self.rootPath = os.path.abspath(rootPath)
        self.reader = reader
        self.dataset, self.class_dic, self.n_samples = self.dataLoader(self.rootPath, self.reader)
        self.shuffle = torch.randperm(self.n_samples)
        self.transform = transform

    # ~ ------

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        global cls_name, cls, img
        for key, value in self.class_dic.items():
            imin = value[1]
            imax = value[2]
            if imin <= idx < imax:
                cls_name = key
                cls = value[0]
                img = self.dataset[cls][idx - imin]
        if self.transform:
            img = self.data_augment(img)
        sample = {'class_name': cls_name, 'class_code': cls, 'image': torch.from_numpy(img)}
        return sample

    # ~ ------
    @classmethod
    def data_augment(cls, array, v_flip=True, h_flip=True):
        """
        array : data to tranform
        v_flip, h_flip : vertical & horizontal flip flags
        """
        will_v_flip, will_h_flip = False, False
        if v_flip and random.random() < 0.5:
            will_v_flip = True
        if h_flip and random.random() < 0.5:
            will_h_flip = True

        im = np.copy(array)
        if will_v_flip:
            im = im[:, ::-1, :]
        if will_h_flip:
            im = im[:, :, ::-1]
        out = np.copy(im)
        return out

    def dataLoader(self, path, reader):
        """
        returns :
        1) The dataset (list): dataset[class][img_index]
        2) A dictionnary {path to each class dir (tif format) or file (hdf5 format): [class, i_min, i_max]}
        3) The total number of images in the dataset
        """
        class_dic = {}
        dataset = []
        n_samples = 0
        count = 0
        if reader == 'hdf5_reader':
            # whole training set for 1 class --> 1 hdf5 file
            file_list = glob.glob(path + '/*.h5')
            for f in file_list:
                data, n_img = hdf5_reader(f)
                class_dic[f] = [count, n_samples, n_samples + n_img]
                dataset.append(data)
                n_samples += n_img
                count += 1
        elif reader == 'tif_reader':
            # whole training set for 1 class --> 1 tif directory
            list_subdir = os.listdir(path)
            # Number of iterations = number of classes
            for subdir in list_subdir:
                ids = self.ids[subdir][0]
                data, n_img = tif_reader(os.path.join(path, subdir), ids)
                class_dic[subdir] = [count, n_samples, n_samples + n_img]
                dataset.append(data)
                n_samples += n_img
                count += 1
        return dataset, class_dic, n_samples

