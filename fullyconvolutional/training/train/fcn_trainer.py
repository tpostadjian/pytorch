import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np


class Trainer():

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
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        # ~ self.params, self.gradParams = model.parameters()
        self.params = model.parameters()
        self.loss = 10000

    def runEpoch(self):
        for batch_idx, (data, target) in enumerate(tqdm(self.dataset)):
            data = Variable(data)
            target = Variable(target)
            if self.mode == 'cuda':
                data = data.cuda()
                target = target.long().cuda()
            output = self.model.forward(data)
            self.loss = self.criterion(output.float(), target)
            self.loss.backward()
            self.optimizer.step()

    # def train(self, epochs):
    #     losses = np.zeros(epochs)
    #     mean_losses = np.zeros(epochs)
    #     it = 0
    #     for e in tqdm(range(1, epochs+1)):
    #         self.runEpoch()
    #         losses[it] = self.loss.data[0]
    #         mean_losses[it] = np.mean(losses[max(0, it-100):it])
