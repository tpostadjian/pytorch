from training.train.fcn_dataset_loader import SPOT_dataset
from training.train.fcn_trainer import Trainer
from training.train.fcn_tester import Tester
from models.fcn_net import fcn
from glob import glob as glob
import random
import torch
import torch.nn as nn

WINDOW_SIZE = (128, 128)
STRIDE = 64
TRAIN_RATIO = 0.8
N_EPOCHS = 500
CLASSES = ['Buildings', 'Roads', 'Vegetation', 'Crop', 'Water', 'Unknown']
CLASSES_WEIGHT = torch.ones(len(CLASSES)).cuda()

data_dir = '/media/tpostadjian/Data/These/Test/data/finistere1/tif/tile_{}.tif'
label_dir = 'training/training_set_generation/label/tile_{}.tif'
# label_dir = 'training/training_set_generation/training_set/label/label_{}.tif'
# data_dir = 'training/training_set_generation/training_set/data/data_{}.tif'
all_files = glob(label_dir.replace('{}', '*'))
all_ids = [f.split('tile_')[-1].split('.')[0] for f in all_files]

train_ids = random.sample(all_ids, int(TRAIN_RATIO*len(all_ids)))
test_ids = list(set(all_ids) - set(train_ids))
print(len(train_ids))

net = fcn(4, 6)

train_dataset = SPOT_dataset(train_ids, data_dir, label_dir, WINDOW_SIZE, cache=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=150)
trainer = Trainer(net, nn.CrossEntropyLoss(weight=CLASSES_WEIGHT, ignore_index=0), train_loader)

tester = Tester(net, test_ids, data_dir, label_dir, WINDOW_SIZE, STRIDE)
# training = trainer.train(N_EPOCHS)

testing = tester.test(5)
