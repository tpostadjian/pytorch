from dataset import ImageDataset, hdf5_reader
import torch.utils.data as data
from net_builder import *
from train import Trainer
import torch.nn as nn

classes = ['bati', 'foret', 'route', 'culture', 'eau']
dataset = ImageDataset('E:/Tristan/Data/finistere/training/tile_15000_20000', reader='tif_reader')
data_loader = data.DataLoader(dataset=dataset, batch_size=16, shuffle=True)

net = Model(make_layers(cfg['4l'], batch_norm=True), 5)
tr = Trainer(data_loader, net, nn.CrossEntropyLoss())

def train(epochs):
    # losses = np.zeros(epochs)
    # mean_losses = np.zeros(epochs)
    it = 0
    for e in range(1, epochs + 1):
        tr.runEpoch()
        # losses[it] = trainer.loss.data[0]
        # mean_losses[it] = np.mean(losses[max(0, it-100):it])

        print('Train Epoch: {} [Loss: {:.6f}]'.format( \
                e, tr.loss.data[0]))
        # if e % 1 == 0:
        #     testing = tr.test(5)
        it += 1

train(10)
