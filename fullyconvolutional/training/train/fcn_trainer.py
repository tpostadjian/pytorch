import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np


class Trainer:

    def __init__(self, model, criterion, dataset, batchSize=200, mode='cuda'):
        super(Trainer, self).__init__()
        self.model = model
        self.mode = mode
        if self.mode == 'cuda':
            self.model = model.cuda()
        self.criterion = criterion
        self.dataset = dataset
        # ~ self.batchSize = args.batchSize
        self.batchSize = batchSize
        # ~ self.params, self.gradParams = model.parameters()
        self.params = model.parameters()
        self.optimizer = optim.SGD(self.params, lr=0.01, momentum=0.9)
        self.avg_loss = 10000

    def runEpoch(self):
        loss_list = np.zeros(10000)
        for id_, (data, target) in enumerate(tqdm(self.dataset)):
            data = Variable(data)
            target = Variable(target)
            if self.mode == 'cuda':
                data = data.cuda()
                target = target.long().cuda()
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output.float(), target)
            loss.backward()
            self.optimizer.step()

            loss_list[id_] = loss.item()
            del(data, target, loss)
        self.avg_loss = np.mean(loss_list[np.nonzero(loss_list)])