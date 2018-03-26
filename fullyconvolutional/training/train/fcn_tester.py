import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix
import gdal
import itertools
import torch

class Tester():

    def __init__(self, model, test_ids, data_img, label_img, window_size, stride, batchSize=200, mode='cuda'):
        super(Tester, self).__init__()
        self.model = model
        self.data_files = [data_img.format(id) for id in test_ids]
        self.label_files = [label_img.format(id) for id in test_ids]
        self.window_size = window_size
        self.stride = stride
        # ~ self.batchSize = args.batchSize
        self.batchSize = batchSize
        self.mode = mode
        if self.mode == 'cuda':
            self.model = model.cuda()
        # ~ self.params, self.gradParams = model.parameters()
        self.params = model.parameters()

    def sliding_window(self, img):
        for l in range(0, img.shape[0], self.stride):
            if l + self.window_size[0] > img.shape[0]:
                l = img.shape[0] - self.window_size[0]
            for c in range(0, img.shape[0], self.stride):
                if c + self.window_size[1] > img.shape[1]:
                    c = img.shape[1] - self.window_size[1]
                yield l, c, self.window_size[0], self.window_size[1]

    def count_sliding_window(self, img):
        """ Count the number of windows in an image """
        count = 0
        for l in range(0, img.shape[0], self.stride):
            if l + self.window_size[0] > img.shape[0]:
                l = img.shape[0] - self.window_size[0]
            for c in range(0, img.shape[1], self.stride):
                if c + self.window_size[1] > img.shape[1]:
                    c = img.shape[1] - self.window_size[1]
                count += 1
        return count

    def grouper(self, n, iterable):
        """ Browse an iterator by chunk of n elements """
        it = iter(iterable)
        while True:
            chunk = tuple(itertools.islice(it, n))
            if not chunk:
                return
            yield chunk

    def test(self, n_classes):
        test_imgs = (1/2**16 * gdal.Open(f).ReadAsArray() for f in self.data_files)
        gt = (gdal.Open(f).ReadAsArray() for f in self.label_files)

        for data, target in zip(test_imgs, gt):
            data=data.transpose(2,1,0)
            pred = np.zeros(data.shape[:2]+(n_classes,))

            for i, coords in enumerate(self.grouper(150, self.sliding_window(data))):

                patches = [np.copy(data[l:l+w, c:c+h]).transpose(2,0,1) for l,c,w,h in coords]
                patches = np.asarray(patches, dtype="float32")
                print(patches.shape)
                patches = Variable(torch.from_numpy(patches).cuda())
                output = self.model(patches)
                output = output.data.cpu().numpy()

