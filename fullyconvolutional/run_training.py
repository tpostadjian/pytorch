from fullyconvolutional.training.train.fcn_dataset_loader import SPOT_dataset
from fullyconvolutional.training.train.fcn_trainer import Trainer
from fullyconvolutional.training.train.fcn_tester import Tester
from fullyconvolutional.models.fcn_net import fcn
from glob import glob as glob
import random
import torch
import torch.utils.data as data
import torch.nn as nn

WINDOW_SIZE = (128, 128)
STRIDE = 64
TRAIN_RATIO = 0.9
N_EPOCHS = 150
CLASSES = ['Buildings', 'Vegetation', 'Water', 'Crop', 'Roads']
# CLASSES = ['Unknown', 'Buildings', 'Vegetation', 'Water', 'Crop', 'Roads']
# weights = [0, 1, 0.4, 0.7, 0.1, 0.8]
# CLASSES_WEIGHT = torch.FloatTensor(weights).cuda()
# CLASSES_WEIGHT = torch.ones(len(CLASSES)).cuda()
CLASSES_WEIGHT = torch.ones(6).cuda()

data_dir = '../../../Data/finistere/img_rescaled/tile_{}.tif'
label_dir = '../../../Data/finistere/label/tile_{}.tif'
# label_dir = 'training/training_set_generation/training_set/label/label_{}.tif'
# data_dir = 'training/training_set_generation/training_set/data/data_{}.tif'
all_files = glob(label_dir.replace('{}', '*'))
all_ids = [f.split('tile_')[-1].split('.')[0] for f in all_files]

train_ids = random.sample(all_ids, int(TRAIN_RATIO * len(all_ids)))
test_ids = list(set(all_ids) - set(train_ids))
print(len(train_ids))

net = fcn(4, 6)
print(net)

train_dataset = SPOT_dataset(train_ids, data_dir, label_dir, WINDOW_SIZE, cache=True)

train_loader = data.DataLoader(train_dataset, batch_size=64, shuffle=True)
tr = Trainer(net, nn.CrossEntropyLoss(weight=CLASSES_WEIGHT, ignore_index=0), train_loader, mode='cuda')
# trainer = Trainer(net, nn.NLLLoss2d(weight=CLASSES_WEIGHT, ignore_index=0), train_loader)

te = Tester(net, nn.CrossEntropyLoss(weight=CLASSES_WEIGHT, ignore_index=0), test_ids, data_dir, label_dir, WINDOW_SIZE, STRIDE)


def train(epochs):

    OUT_DIR = '.'
    best_acc = 0
    # Some training perfomances
    LOSS_TRAIN_FILE = OUT_DIR + '/train_losses.txt'
    LOSS_TEST_FILE = OUT_DIR + '/test_losses.txt'
    ACC_TEST_FILE = OUT_DIR + '/test_acc.txt'

    print('Initial best accuracy: {:.2f}'.format(best_acc))
    with open(LOSS_TRAIN_FILE, 'w') as f_trainloss, \
            open(LOSS_TEST_FILE, 'w') as f_testloss, \
            open(ACC_TEST_FILE, 'w') as f_testacc:

        for e in range(epochs):
            # Training
            print('\n----------------------------')
            print('Epoch: {}'.format(e))
            tr.runEpoch()
            f_trainloss.write('{:.2f}\n'.format(tr.avg_loss))
            print('\nTraining loss: {:.2f}'.format(tr.avg_loss))
            if e % 5 == 0:
                te.test(5)
            torch.save(net.state_dict(), './net')
    f_trainloss.close()


def save_state(state, is_best, out_dir):
    torch.save(state, out_dir + '/model_state.pth')
    if is_best:
        torch.save(state, out_dir + '/model_best.pth')
        torch.save(net.state_dict(), './net')

train(N_EPOCHS)
