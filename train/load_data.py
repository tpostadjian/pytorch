from dataset import ImageDataset, hdf5_reader
from net_builder import *
from train import Trainer
import numpy as np

classes = ['bati', 'foret', 'route', 'culture', 'eau']

ds = ImageDataset('/media/tpostadjian/Data/These/Test/data/brest_2016/Images/training/')

it = ds.trainGenerator()

net = Model(make_layers(cfg['4l']),5)

tr = Trainer(net, nn.CrossEntropyLoss(), ds)
tr.runEpoch()

#Nouveau Commit