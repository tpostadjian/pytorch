import torch.nn as nn
import torch.nn.functional as F

class fcn(nn.Module):

    def __init__(self, n_bands, n_classes):
        super(fcn, self).__init__()
        self.pool = nn.MaxPool2d(2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2)

        self.conv1 = nn.Conv2d(n_bands, 32, 3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4_bn = nn.BatchNorm2d(256)

        self.conv4up = nn.Conv2d(256, 128, 3, padding=1)
        self.conv4up_bn = nn.BatchNorm2d(128)
        self.conv3up = nn.Conv2d(128, 64, 3, padding=1)
        self.conv3up_bn = nn.BatchNorm2d(64)
        self.conv2up = nn.Conv2d(64, 32, 3, padding=1)
        self.conv2up_bn = nn.BatchNorm2d(32)
        self.conv1up = nn.Conv2d(32, n_classes, 3, padding=1)

    def forward(self, batch):
        # encoder
        size_verbose = False
        x = batch
        if size_verbose:
            print("input : "+str(x.size()))
        x = self.conv1_bn(F.relu(self.conv1(x)))
        x, mask1 = self.pool(x)
        if size_verbose:
            print("1st : "+str(x.size()))

        x = self.conv2_bn(F.relu(self.conv2(x)))
        x, mask2 = self.pool(x)
        if size_verbose:
            print("2nd : "+str(x.size()))

        x = self.conv3_bn(F.relu(self.conv3(x)))
        x, mask3 = self.pool(x)
        if size_verbose:
            print("3rd : "+str(x.size()))

        x = self.conv4_bn(F.relu(self.conv4(x)))
        x, mask4 = self.pool(x)
        if size_verbose:
            print("4th : "+str(x.size()))

        # decoder
        x = self.unpool(x, mask4)
        if size_verbose:
            print("4th unpool : "+str(x.size()))
        x = self.conv4up_bn(F.relu(self.conv4up(x)))

        x = self.unpool(x, mask3)
        if size_verbose:
            print("3rd unpool : " + str(x.size()))
        x = self.conv3up_bn(F.relu(self.conv3up(x)))

        x = self.unpool(x, mask2)
        if size_verbose:
            print("2nd unpool : " + str(x.size()))
        x = self.conv2up_bn(F.relu(self.conv2up(x)))

        x = self.unpool(x, mask1)
        if size_verbose:
            print("1st unpool : " + str(x.size()))
        x = F.relu(self.conv1up(x))

        x = F.log_softmax(x, dim=1)

        return x