import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np


class Trainer():

    def __init__(self, data_loader, model, criterion, batchsize, mode='cuda'):
        super(Trainer, self).__init__()
        self.data_loader = data_loader
        self.model = model
        self.mode = mode
        if self.mode == 'cuda':
            self.model = model.cuda()
        self.criterion = criterion
        self.batchsize = batchsize
        self.params = model.parameters()
        self.optimizer = optim.SGD(self.params, lr=0.00000001, momentum=0.9)
        self.avg_loss = 10000

    def runEpoch(self):
        loss_list = np.zeros(10000)
        for it_, batch in enumerate(tqdm(self.data_loader)):
            data = Variable(batch['image'])
            target = Variable(batch['class_code'])
            # forward
            if self.mode == 'cuda':
                data = data.cuda()
                target = target.long().cuda()
            output = self.model.forward(data)
            loss = self.criterion(output.float(), target)
            loss.backward()
            self.optimizer.step()
            loss_list[it_] = loss.item()
            self.avg_loss = np.mean(loss_list[max(0, it_-self.batchsize):it_])
