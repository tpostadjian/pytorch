import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


class Trainer():

    def __init__(self, model, criterion, dataset, batchSize=200, mode='cuda'):
        super(Trainer, self).__init__()
        self.model = model
        if mode == 'cuda':
            self.model = model.cuda()
        self.criterion = criterion
        self.dataset = dataset
        # ~ self.batchSize = args.batchSize
        self.batchSize = batchSize
        self.optimizer = optim.SGD
        # ~ self.params, self.gradParams = model.parameters()
        self.params = model.parameters()

    def runEpoch(self):
        iterator = self.dataset.trainGenerator(self.batchSize)
        for data, target in iterator:
            # forward
            data = Variable(data).cuda()
            target = Variable(target).long().cuda()
            output = self.model.forward(data)
            loss = self.criterion(output, target)
            loss.backward()
