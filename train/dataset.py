import torch
import torch.utils.data as data
import numpy as np
import h5py
import os
import glob as glob


def hdf5_reader(h5_file):
    h5_ds = h5py.File(h5_file, 'r')
    imgBuffer = []
    img_count = 1
    eof = False
    while eof == False:
        img_str = 'img_' + str(img_count)
        try:
            img = h5_ds[img_str]
            img = np.array(img)
            img = torch.from_numpy(img)
            img = img.float()
            imgBuffer.append(img)
            img_count += 1
        except KeyError:
            eof = True
            img_count -= 1
            pass
    return imgBuffer, img_count


class ImageDataset(data.Dataset):
    # ~ ------
    def __init__(self, rootPath, trainRatio=1., reader=hdf5_reader, transform=None):
        """
        Args:
            rootPath (string): Path to the h5 files containing image dataset
            trainRatio : used to further split between training/validating datasets
            reader : Input file reader
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.rootPath = os.path.abspath(rootPath)
        self.reader = reader
        self.dataset, self.class_dic, self.n_samples = self.dataLoader(self.rootPath, self.reader)
        self.shuffle = torch.randperm(self.n_samples)
        self.n_train = int(self.n_samples * trainRatio)
        self.n_valid = 1 - self.n_train
        self.transform = transform

    # ~ ------

    # ~ ------
    def batchGenerator(self, n, class_dic, dataset, batchsize=200):

        global Y

        def getClassImg(idx, dic, dataset):
            for key, value in dic.items():
                imin = value[1]
                imax = value[2]
                if imin <= idx < imax:
                    cls = value[0]
                    img = dataset[cls - 1][idx - imin]
            return cls, img

        """
        Used by both trainGenerator et validGenerator
        """
        shuffle = torch.randperm(n)
        indBatch = shuffle.split(batchsize)

        i = 0
        if i <= len(indBatch):
            currentBatch = indBatch[i]
            imgBatch = []
            clsBatch = []
            l = len(currentBatch)  # current batch length
            for j in range(l):
                sample_ind = currentBatch[j]
                cls, img = getClassImg(sample_ind, class_dic, dataset)
                imgBatch.append(img)
                clsBatch.append(cls)

            X = torch.Tensor(l, 4, 65, 65)
            Y = torch.Tensor(l)
            for s in range(l):
                X[s] = imgBatch[s]
                Y[s] = clsBatch[s]
            i += 1
        yield X, Y
        # ~ ------

    def trainGenerator(self, batchsize=200):

        return self.batchGenerator(self.n_train, self.class_dic, self.dataset, batchsize)

    # ~ ------

    # ~ ------
    def dataLoader(self, path, reader):
        """
        returns :
        1) A dictionnary {path to each h5 file : [class, i_min, i_max]}
        2) The total number of images in the dataset
        """
        class_dic = {}
        dataset = []
        n_samples = 0
        file_list = glob.glob(path + '/*.h5')
        count = 1
        for f in file_list:
            imgBuffer, n_img = reader(f)
            class_dic[f] = [count, n_samples, n_samples + n_img]
            dataset.append(imgBuffer)
            n_samples += n_img
            count += 1
        return dataset, class_dic, n_samples
