import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix
import gdal
import itertools
import torch
import matplotlib.pyplot as plt


def convert_to_color(arr_2d):
    """ Numeric labels to RGB-color encoding """
    palette = {0: (0, 0 , 0),
               1: (255, 0, 0),      # Buildings (red)
               2: (0, 255, 0),      # Vegetation (green)
               3: (0, 0, 255),      # Water (Blue)
               4: (255, 255, 0),    # Crop (yellow)
               5: (100, 100, 100)}  # Road (Grey)

    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d


def grouper(n, iterable):
    """ Browse an iterator by chunk of n elements """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def metrics(predictions, gts, label_values):
    cm = confusion_matrix(
        gts,
        predictions,
        range(len(label_values)))

    print("Confusion matrix :")
    print(cm)

    print("---")

    # Compute global accuracy
    total = sum(sum(cm))
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    print("{} pixels processed".format(total))
    print("Total accuracy : {}%".format(accuracy))

    print("---")

    # Compute F1 score
    F1Score = np.zeros(len(label_values))
    for i in range(len(label_values)):
        try:
            F1Score[i] = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except:
            # Ignore exception if there is no element in class i for test set
            pass
    print("F1Score :")
    for l_id, score in enumerate(F1Score):
        print("{}: {}".format(label_values[l_id], score))

    print("---")

    # Compute kappa coefficient
    total = np.sum(cm)
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)
    kappa = (pa - pe) / (1 - pe)
    print("Kappa: " + str(kappa))
    return accuracy

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

    def test(self, n_classes):
        test_imgs = (1/2**16 * gdal.Open(f).ReadAsArray() for f in self.data_files)
        gt = (gdal.Open(f).ReadAsArray() for f in self.label_files)

        all_preds = []
        all_gts = []

        it = 0
        for data, target in zip(test_imgs, gt):
            data = data.transpose(2,1,0)
            pred = np.zeros(data.shape[:2]+(n_classes+1,))

            for i, coords in enumerate(grouper(150, self.sliding_window(data))):

                patches = [np.copy(data[l:l+w, c:c+h]).transpose(2,0,1) for l,c,w,h in coords]
                patches = np.asarray(patches, dtype="float32")
                patches = Variable(torch.from_numpy(patches).cuda())

                output = self.model(patches)
                output = output.data.cpu().numpy()

                for out, (l,c,w,h) in zip(output, coords):
                    out = out.transpose(1,2,0)
                    pred[l:l+w, c:c+h] += out
                del output

            pred = np.argmax(pred, axis=-1)

            # Display the result
            if it % 25 == 0:
                print(self.data_files[it])
                fig = plt.figure()
                fig.add_subplot(1, 3, 1)
                plt.imshow(np.asarray(2**16 * data.transpose(1,0,2), dtype='uint8'))
                fig.add_subplot(1, 3, 2)
                plt.imshow(convert_to_color(pred).transpose(1,0,2))
                fig.add_subplot(1, 3, 3)
                plt.imshow(convert_to_color(target))
                plt.show()

            all_preds.append(pred)
            all_gts.append(target)

            cls = ['Unknown', 'Buildings', 'Vegetation', 'Water', 'Crop', 'Roads']
            metrics(pred.transpose(1,0).ravel(), target.ravel(), cls)
            accuracy = metrics(np.concatenate([p.ravel() for p in all_preds]),
                               np.concatenate([p.ravel() for p in all_gts]).ravel(), cls)

            it += 1
            if all:
                return accuracy, all_preds, all_gts
            else:
                return accuracy

