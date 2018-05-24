import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

class Trainer():

    def __init__(self, data_loader, model, criterion, mode='cuda'):
        super(Trainer, self).__init__()
        self.data_loader = data_loader
        self.model = model
        self.mode = mode
        if self.mode == 'cuda':
            self.model = model.cuda()
        self.criterion = criterion
        self.params = model.parameters()
        self.optimizer = optim.SGD(self.params, lr=0.01)
        self.loss = 10000

    def runEpoch(self):
        for it, batch in enumerate(tqdm(self.data_loader)):
            data, target = batch
            data = Variable(data)
            target = Variable(target)
            # forward
            if self.mode == 'cuda':
                data = data.cuda()
                target = target.long().cuda()
            output = self.model.forward(data)
            self.loss = self.criterion(output.float(), target)
            self.loss.backward()
            self.optimizer.step()
