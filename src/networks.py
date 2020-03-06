from torch import nn

from .activations import Swish, Mish

class LeNet(nn.Module):
    def __init__(self, hparams):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,  out_channels=20, kernel_size=5,
                               stride=1, padding=2, dilation=1,
                               padding_mode='zeros')
        self.pool1 = nn.MaxPool2d(kernel_size=2,
                                  stride=2, padding=0, dilation=1,
                                  ceil_mode=True)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5,
                               stride=1, padding=2, dilation=1,
                               padding_mode='zeros')
        self.pool2 = nn.MaxPool2d(kernel_size=2,
                                  stride=2, padding=0, dilation=1,
                                  ceil_mode=True)
        self.fc1   = nn.Linear(7*7*50, 500)
        self.fc2   = nn.Linear(500, 10)
        self.softmax = nn.LogSoftmax(dim=1)

        if   hparams.activation.lower() == 'relu':
            self.act = nn.ReLU()
        elif hparams.activation.lower() == 'swish':
            self.act = Swish()
        elif hparams.activation.lower() == 'mish':
            self.act = Mish()
        else:
            raise NotImplementedError(f'Unknown activation "{hparams.activation}"')

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
