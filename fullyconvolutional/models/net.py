import torch.nn as nn

class fcn(nn.Module):

    def __init__(self, n_bands, n_classes):
        super(fcn, self).__init__()
        self.pool = nn.MaxPool2d(2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2)

        self.conv1 = nn.conv2d(n_bands, 32, 3)
        self.conv2 = nn.conv2d(32, 64, 3)
        self.conv3 = nn.conv2d(64, 128, 3)
        self.conv4 = nn.conv2d(128, 256, 3)

        self.conv4up = nn.conv2d(256, 128, 3)
        self.conv3up = nn.conv2d(128, 64, 3)
        self.conv2up = nn.conv2d(64, 32, 3)
        self.conv1up = nn.conv2d(32, n_classes, 3)

    def forward(self, batch):
        # encoder
        x = batch
        x = nn.ReLU(self.conv1(x))
        x, mask1 = self.pool(x)

        x = nn.ReLU(self.conv2(x))
        x, mask2 = self.pool(x)

        x = nn.ReLU(self.conv3(x))
        x, mask3 = self.pool(x)

        x = nn.ReLU(self.conv4(x))
        x, mask4 = self.pool(x)

        # decoder
        x = self.unpool(x, mask4)
        x = nn.ReLU(self.conv4up(x))

        x = self.unpool(x, mask3)
        x = nn.ReLU(self.conv3up(x))

        x = self.unpool(x, mask2)
        x = nn.ReLU(self.conv2up(x))

        x = self.unpool(x, mask1)
        x = nn.ReLU(self.conv1up(x))

        x = nn.log_softmax(x)
        return x