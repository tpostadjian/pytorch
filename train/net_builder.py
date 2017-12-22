import torch
import torch.nn as nn
import math


class Model(nn.Module):

    def __init__(self, features, n_classes):
        super(Model, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, n_classes)
        )
        self._initialize_weights()

    def forward(self, x):

        ## show output size at each layer
        # ~ for i in range(12):
        # ~ x = self.features[i](x)
        # ~ print(x.size())
        # ~ print(self.features[0])
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 4
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    '3l': [32, 'M', 64, 'M', 128, 'M'],
    '4l': [16, 'M', 32, 'M', 64, 'M', 128, 'M'],
    '5l': [32, 'M', 64, 64, 'M', 128, 'M', 256, 'M'],
    '6l': [32, 'M', 64, 64, 'M', 128, 128, 'M', 256, 'M']
}
