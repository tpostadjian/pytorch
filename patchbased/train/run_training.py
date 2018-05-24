from train_valid_split import split_dataset
from dataset import ImageDataset
import torch.utils.data as data
from net_builder import *
from trainer import Trainer
import torch.nn as nn

TRAIN_RATIO = 0.85
CLASSES = ['bati', 'foret', 'route', 'culture', 'eau']
TRAIN_DIR = 'E:/Tristan/Data/finistere/training/tile_15000_20000'

train_ids, valid_ids = split_dataset(TRAIN_DIR, TRAIN_RATIO)

train_dataset = ImageDataset(train_ids, TRAIN_DIR, reader='tif_reader')
train_loader = data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

valid_dataset = ImageDataset(valid_ids, TRAIN_DIR, reader='tif_reader')
train_loader = data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

net = Model(make_layers(cfg['4l'], batch_norm=True), 5)
tr = Trainer(train_loader, net, nn.CrossEntropyLoss())

#
def train(epochs):
    # losses = np.zeros(epochs)
    # mean_losses = np.zeros(epochs)
    it = 0
    for e in range(1, epochs + 1):
        tr.runEpoch()
        # losses[it] = trainer.loss.data[0]
        # mean_losses[it] = np.mean(losses[max(0, it-100):it])

        print('Train Epoch: {} [Loss: {:.6f}]'.format(e, tr.loss.item()))
        # if e % 1 == 0:
        #     testing = tr.test(5)
        it += 1


train(10)
