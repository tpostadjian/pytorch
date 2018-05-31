import torch
import torch.optim as optim
import torch.utils.data as data

from dataset import ImageDataset
from net_builder import *
from tester import Tester
from train_valid_split import split_dataset
from trainer import Trainer

# import numpy as np
# import matplotlib.pyplot as plt
import os, time
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('data', type=str, help='path to dataset')
parser.add_argument('--outdir', type=str, default='./results', help='path to saving directory')
parser.add_argument('--ratio', type=float, default=0.9, help='train/validation ratio')
parser.add_argument('--classes', nargs='+', default=['bati', 'culture', 'eau', 'foret', 'route'], help='list of input classes')
parser.add_argument('--epochs', type=int, default=150, help='number of epochs')
parser.add_argument('--cuda', type=str, default='CUDA', help='for GPU computation')
parser.add_argument('--resume', type=str, default='', help='path to existing state model for resuming training')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weightdecay', type=float, default=1e-4, help='weight decay')
args = parser.parse_args()


def main(args):
    # TRAIN_RATIO = 0.007
    # CLASSES = ['bati', 'culture', 'eau', 'foret', 'route']
    # TRAIN_DIR = 'E:/Tristan/Data/finistere/training_dataset_rescaled'
    # EPOCHS = 100
    # DIR_PERF = './zero_grad'
    # LOSS_TRAIN_FILE = DIR_PERF+'/train_losses.txt'
    # LOSS_TEST_FILE = DIR_PERF+'/test_losses.txt'
    # ACC_TEST_FILE = DIR_PERF+'/test_acc.txt'
    # mode = 'CUDA'
    # RESUME = False
    # RESUME_STATE = DIR_PERF+'/model_best.pth'
    TRAIN_DIR = args.data
    OUT_DIR = args.outdir
    TRAIN_RATIO = args.ratio
    CLASSES = args.classes
    EPOCHS = args.epochs
    mode = args.cuda
    RESUME = args.resume

    try:
        os.makedirs(OUT_DIR)
    except OSError:
        pass

    train_ids, test_ids, n_train, n_test = split_dataset(TRAIN_DIR, TRAIN_RATIO)

    # Loading training dataset
    start = time.clock()
    train_dataset = ImageDataset(train_ids, TRAIN_DIR, reader='tif_reader')
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
    c_time = time.clock() - start
    print('Training set loaded: ' + str(n_train) + ' samples in %.2f sec' % c_time)

    # Loading evaluation dataset
    start = time.clock()
    test_dataset = ImageDataset(test_ids, TRAIN_DIR, reader='tif_reader')
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)
    c_time = time.clock() - start
    print('Test set loaded: ' + str(n_test) + ' samples in %.2f sec' % c_time)

    # Net definition + cuda check
    net = Model(make_layers(cfg['4l'], batch_norm=True), 5)
    if mode == 'CUDA':
        net.cuda()

    # Loss function and optimizer definition
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weightdecay)
    start_epoch = 0
    best_acc = 0

    # Case where using pre-trained network
    if RESUME:
        print('resuming!')
        if os.path.isfile(RESUME):
            print('Loading model state: {}'.format(RESUME))
            check_state = torch.load(RESUME)
            start_epoch = check_state['epoch']
            best_acc += check_state['best_acc']
            net.load_state_dict(check_state['params'])
            optimizer.load_state_dict(check_state['optimizer'])
            print("Resuming at epoch {}".format(start_epoch))
        else:
            print("No pre-trained net found at '{}'".format(RESUME))

    print(net)
    tr = Trainer(train_loader, net, criterion, optimizer)
    te = Tester(test_loader, net, criterion, CLASSES)

    # Some training perfomances
    LOSS_TRAIN_FILE = OUT_DIR+'/train_losses.txt'
    LOSS_TEST_FILE = OUT_DIR+'/test_losses.txt'
    ACC_TEST_FILE = OUT_DIR+'/test_acc.txt'

    print('Initial best accuracy: {:.2f}'.format(best_acc))
    with open(LOSS_TRAIN_FILE, 'w') as f_trainloss, \
            open(LOSS_TEST_FILE, 'w') as f_testloss, \
            open(ACC_TEST_FILE, 'w') as f_testacc:

        for e in range(start_epoch, EPOCHS):

            # Training
            print('\n----------------------------')
            print('Epoch: {}'.format(e+1))
            net.train()
            tr.runEpoch()
            f_trainloss.write('{:.2f}\n'.format(tr.avg_loss))
            print('\nTraining loss: {:.2f}'.format(tr.avg_loss))

            # Evaluation
            net.eval()
            te.test()
            accuracy = te.accuracy
            f_testloss.write('{:.2f}\n'.format(te.avg_loss))
            f_testacc.write('{:.2f}\n'.format(accuracy))
            # Save best_acc and current model state
            is_best_acc = accuracy > best_acc
            best_score = max(accuracy, best_acc)
            state = {
                'epoch': e+1,
                'best_acc': best_score,
                'params': net.float().state_dict(),
                'optimizer': tr.optimizer.state_dict()
            }
            save_state(state, is_best_acc, OUT_DIR)

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
    f_trainloss.close()
    f_testloss.close()
    f_testacc.close()


def save_state(state, is_best, out_dir):
    torch.save(state, out_dir+'/model_state.pth')
    if is_best:
        torch.save(state, out_dir+'/model_best.pth')

main(args)
