from torch.autograd import Variable
from tqdm import tqdm
import numpy as np


class Trainer():

    def __init__(self, data_loader, model, criterion, optimizer, mode='cuda'):
        super(Trainer, self).__init__()
        self.data_loader = data_loader
        self.model = model
        self.criterion = criterion
        self.mode = mode
        # self.params = model.parameters()
        self.optimizer = optimizer
        self.avg_loss = 10000

    def runEpoch(self):
        loss_list = np.zeros(100000)
        for it, batch in enumerate(tqdm(self.data_loader)):
            self.optimizer.zero_grad()
            data = Variable(batch['image'])
            target = Variable(batch['class_code'])
            # forward
            if self.mode == 'cuda':
                data = data.cuda()
                target = target.long().cuda()
            output = self.model(data)
            loss = self.criterion(output.float(), target)
            loss.backward()
            self.optimizer.step()
            loss_list[it] = loss.item()
        self.avg_loss = np.mean(loss_list[np.nonzero(loss_list)])
