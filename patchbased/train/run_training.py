import torch
import torch.utils.data as data
import torch.nn as nn
from train_valid_split import split_dataset
from dataset import ImageDataset
from net_builder import *
from trainer import Trainer
from tester import Tester

import numpy as np
import matplotlib.pyplot as plt
import time

TRAIN_RATIO = 0.007
CLASSES = ['bati', 'culture', 'eau', 'foret', 'route']
TRAIN_DIR = 'E:/Tristan/Data/finistere/training_dataset_rescaled'

train_ids, test_ids, n_train, n_test = split_dataset(TRAIN_DIR, TRAIN_RATIO)

start = time.clock()
train_dataset = ImageDataset(train_ids, TRAIN_DIR, reader='tif_reader')
train_loader = data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
c_time = time.clock() - start
print('Training set loaded: '+str(n_train)+' samples in %.2f sec' % c_time)

start = time.clock()
test_dataset = ImageDataset(test_ids, TRAIN_DIR, reader='tif_reader')
test_loader = data.DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
c_time = time.clock() - start
print('Test set loaded: '+str(n_test)+' samples in %.2f sec' % c_time)

net = Model(make_layers(cfg['4l'], batch_norm=True), 5)
tr = Trainer(train_loader, net, nn.CrossEntropyLoss(), batchsize=16)
te = Tester(test_loader, net)


def train(epochs):
    file = open('losses.txt', 'w')
    for e in range(1, epochs + 1):
        tr.runEpoch()
        file.write('%.2f\n' % tr.avg_loss)
        print('\nTrain Epoch: {} [Loss: {:.6f}]'.format(e+1, tr.avg_loss))
        if e % 100 == 0:
            te.test(5)

            idx = np.random.randint(0, len(test_ids))
            img = test_dataset[idx]['image']
            class_name = test_dataset[idx]['class_name']
            fig = plt.figure()
            fig.suptitle(class_name)
            img *= 255
            img = img.numpy()
            img = img.transpose(1,2,0)
            img = img[:, :, 0:3]
            plt.imshow(np.asarray(img, dtype='uint8'))
            plt.show()

        if e % 150 == 0:
            accuracy = te.test(5, acc_only=True)
            torch.save(net.state_dict(), './net/net_epoch{}_loss_{}'.format(e, accuracy))
    file.close()
train(6000)
