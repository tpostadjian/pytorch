import torch
import torch.optim as optim
import torch.utils.data as data

from patchbased.train.dataset import ImageDataset
from patchbased.train.net_builder import *
from patchbased.train.tester import Tester
from patchbased.train.train_valid_split import split_dataset
from patchbased.train.trainer import Trainer

# import numpy as np
# import matplotlib.pyplot as plt
import os, time

TRAIN_RATIO = 0.0007
CLASSES = ['bati', 'culture', 'eau', 'foret', 'route']
TRAIN_DIR = 'E:/Tristan/Data/finistere/training_dataset_rescaled'
LOSS_TRAIN_FILE = './results/train_losses.txt'
ACC_TEST_FILE = './results/test_acc.txt'
mode = 'CUDA'
RESUME = False
RESUME_STATE = 'results/model_best.pth'

try:
    os.makedirs(os.path.dirname(LOSS_TRAIN_FILE))
except OSError:
    pass

train_ids, test_ids, n_train, n_test = split_dataset(TRAIN_DIR, TRAIN_RATIO)

# Loading training dataset
start = time.clock()
train_dataset = ImageDataset(train_ids, TRAIN_DIR, reader='tif_reader')
train_loader = data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
c_time = time.clock() - start
print('Training set loaded: '+str(n_train)+' samples in %.2f sec' % c_time)

# Loading evaluation dataset
start = time.clock()
test_dataset = ImageDataset(test_ids, TRAIN_DIR, reader='tif_reader')
test_loader = data.DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)
c_time = time.clock() - start
print('Test set loaded: '+str(n_test)+' samples in %.2f sec' % c_time)

# Net definition + cuda check
net = Model(make_layers(cfg['4l'], batch_norm=True), 5)
if mode == 'CUDA':
    net.cuda()

# Loss function and optimizer definition
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),
                      lr=0.001,
                      momentum=0.9,
                      weight_decay=0.0005)
best_acc = 0

# Case where using pre-trained network
if RESUME:
    if os.path.isfile(RESUME_STATE):
        check_state = torch.load(RESUME_STATE)
        resume_epoch = check_state['epoch']
        best_acc += check_state['best_acc']
        net.load_state_dict(check_state['params'])
        optimizer.load_state_dict(check_state['optimizer'])
        print("Resuming at epoch {}".format(resume_epoch+1))
    else:
        print("No pre-trained net found at '{}'".format(RESUME_STATE))


print(net)
tr = Trainer(train_loader, net, criterion, optimizer)
te = Tester(test_loader, net, criterion, CLASSES)


def train(start_epochs, best_score):
    print('Initial best accuracy: {:.2f}'.format(best_score))
    with open(LOSS_TRAIN_FILE, 'w') as f_train, open(ACC_TEST_FILE, 'w') as f_test:
        for e in range(1, start_epochs + 1):

            # Training
            net.train()
            tr.runEpoch()
            f_train.write('{:.2f}\n'.format(tr.avg_loss))
            print('\nTrain Epoch: {} [Loss: {:.2f}]'.format(e+1, tr.avg_loss))

            # Evaluation
            net.eval()
            accuracy = te.test()
            f_test.write('{:.2f}\n'.format(accuracy))

            # Save best_acc and current model state
            is_best_acc = accuracy > best_score
            best_score = max(accuracy, best_score)
            state = {
                'epoch': e,
                'best_acc': best_score,
                'params': net.float().state_dict(),
                'optimizer': tr.optimizer.state_dict()
            }
            save_state(state, is_best_acc)

            # idx = np.random.randint(0, len(test_ids))
            # img = test_dataset[idx]['image']
            # class_name = test_dataset[idx]['class_name']
            # fig = plt.figure()
            # fig.suptitle(class_name)
            # img *= 255
            # img = img.numpy()
            # img = img.transpose(1,2,0)
            # img = img[:, :, 0:3]
            # plt.imshow(np.asarray(img, dtype='uint8'))
            # plt.show()
    f_train.close()
    f_test.close()


def save_state(state, is_best):
    torch.save(state, './results/model_state.pth')
    if is_best:
        torch.save(state, './results/model_best.pth')


train(100, best_acc)
