from torch.autograd import Variable
import numpy as np
from sklearn.metrics import confusion_matrix
import gdal
import itertools
import torch
import matplotlib.pyplot as plt


def convert_to_color(arr_greyscale):
    palette = {#0: (0, 0, 0),
               1: (255, 0, 0),      # Buildings (red)
               2: (0, 255, 0),      # Vegetation (green)
               3: (0, 0, 255),      # Water (Blue)
               4: (255, 255, 0),    # Crop (yellow)
               5: (100, 100, 100)}  # Road (Grey)

    nc, nl = arr_greyscale.shape
    arr_color = np.zeros((nc, nl, 3), dtype=np.uint8)

    for c, i in palette.items():
        b = arr_greyscale == c
        arr_color[b] = i
    return arr_color


def grouper(n, iterable):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def metrics(preds, targets, classes):

    confusion = confusion_matrix(
        targets,
        preds,
        range(len(classes)))
    print("***")

    print("Confusion matrix: ")
    print(confusion)

    print("***")

    # Overall accuracy
    total = sum(sum(confusion))
    accuracy = sum([confusion[x][x] for x in range(len(confusion))])
    accuracy *= 100 / float(total)
    print("Evaluation on {} pixels: ".format(total))
    print("Overall accuracy: {:.2f}%".format(accuracy))

    print("***")

    # Kappa
    total = np.sum(confusion)
    Pa = np.trace(confusion) / float(total)
    Pe = np.sum(np.sum(confusion, axis=0) * np.sum(confusion, axis=1)) / float(total * total)
    kappa = (Pa - Pe) / (1 - Pe)
    print("Kappa: {:.2f}".format(kappa))

    print("***")

    # F1-score / class
    F1Score = np.zeros(len(classes))
    for cls in range(len(classes)):
        try:
            F1Score[cls] = 2.*confusion[cls, cls]/(np.sum(confusion[cls, :])+np.sum(confusion[:, cls]))
        except:
            pass
    print("F1Score: ")
    for cls, score in enumerate(F1Score):
        print("{}: {:.2f}".format(classes[cls], score))

    print("***")

    # IoU = TP/(TP+FN+FP)
    iou = np.zeros(len(classes))
    for cls in range(len(classes)):
        intersection = np.float(confusion[cls, cls])
        union = np.sum(confusion[cls, :])+np.sum(confusion[:, cls])-intersection
        if union == 0:
            iou[cls] = np.nan  # no target / groundtruth for class cls
        else:
            iou[cls] = intersection/union
    print("IoU: ")
    for cls, score in enumerate(iou):
        print("{}: {:.2f}".format(classes[cls], score))

    return accuracy


class Tester:

    def __init__(self, model, criterion, test_ids, data_img, label_img, window_size, stride, batchSize=64, mode='cuda'):
        super(Tester, self).__init__()
        self.model = model
        self.criterion = criterion
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
        self.avg_loss = 10000
        self.accuracy = 0

    def sliding_window(self, img):
        for l in range(0, img.shape[1], self.stride):
            if l + self.window_size[0] > img.shape[1]:
                l = img.shape[1] - self.window_size[0]
            for c in range(0, img.shape[2], self.stride):
                if c + self.window_size[1] > img.shape[2]:
                    c = img.shape[2] - self.window_size[1]
                yield l, c, self.window_size[0], self.window_size[1]

    def count_sliding_window(self, img):
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
        test_imgs = (gdal.Open(f).ReadAsArray() for f in self.data_files)
        gt = (gdal.Open(f).ReadAsArray() for f in self.label_files)

        all_preds = []
        all_gts = []
        loss_list = np.zeros(1000000)

        for data, target in zip(test_imgs, gt):
            # data = data.transpose(1, 2, 0)
            # pred = np.zeros((n_classes+1,) + data.shape[1:])
            pred = np.zeros((n_classes,) + data.shape[1:])
            it = 0

            for i, coords in enumerate(grouper(self.batchSize, self.sliding_window(data))):

                patches = [np.copy(data[:,l:l+w, c:c+h]) for l, c, w, h in coords]
                patches = np.asarray(patches, dtype="float32")
                patches = Variable(torch.from_numpy(patches).cuda())

                output = self.model(patches)
                output = output.data.cpu().numpy()

                for out, (l, c, w, h) in zip(output, coords):
                    # out = out.transpose(1, 2, 0)
                    out = out[1:]
                    pred[:, l:l+w, c:c+h] += out
                del output

            pred = np.argmax(pred, axis=0)

            # # Display the result
            # if it % 25 == 0:
            #     fig = plt.figure()
            #     fig.add_subplot(1, 3, 1)
            #     plt.imshow(np.asarray(data[:3, :, :].transpose(1, 2, 0), dtype='uint8'))
            #     fig.add_subplot(1, 3, 2)
            #     plt.imshow(convert_to_color(pred))
            #     fig.add_subplot(1, 3, 3)
            #     plt.imshow(convert_to_color(target))
            #     plt.show()

            all_preds.append(pred)
            all_gts.append(target)

            cls = ['Unknown', 'Buildings', 'Vegetation', 'Water', 'Crop', 'Roads']
            metrics(pred.transpose(1,0).ravel(), target.ravel(), cls)
            accuracy = metrics(np.concatenate([p.ravel() for p in all_preds]),
                               np.concatenate([p.ravel() for p in all_gts]).ravel(), cls)

            it += 1

            return accuracy

