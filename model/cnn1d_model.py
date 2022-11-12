import torch.nn as nn
import torch.nn.functional as F
import torch

'''
class conv1d_block(nn.Module):
    """
    https://www.kaggle.com/code/super13579/u-net-1d-cnn-with-pytorch/notebook
    """
    def __init__(self, in_layer, out_layer, kernel_size, stride, dilation):
        super(conv1d_block, self).__init__()

        self.conv1 = nn.Conv1d(in_layer, out_layer, kernel_size=kernel_size, stride=stride, dilation=dilation,
                               padding=3, bias=True)
        self.bn = nn.BatchNorm1d(out_layer)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        out = self.relu(x)
'''


class conv1d_block(nn.Module):
    def __init__(self):
        super(conv1d_block, self).__init__()

        self.conv1 = nn.Sequential(
            torch.nn.Conv1d(1, 8,
                            kernel_size=5, stride=2, padding=0),
            torch.nn.Dropout(p=0.25),
            torch.nn.LeakyReLU(negative_slope=0.1),
        )
        self.conv2 = nn.Sequential(
            torch.nn.Conv1d(8, 16,
                            kernel_size=7, stride=3, padding=0),
            torch.nn.Dropout(p=0.25),
            torch.nn.LeakyReLU(negative_slope=0.1),
        )
        self.conv3 = nn.Sequential(
            torch.nn.Conv1d(16, 32,
                            kernel_size=11, stride=5, padding=0),
            torch.nn.Dropout(p=0.25),
            torch.nn.LeakyReLU(negative_slope=0.1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        out = self.conv3(x)
        return out


class conv1d_model(nn.Module):
    def __init__(self):
        super(conv1d_model, self).__init__()
        self.conv = conv1d_block()
        # First fully connected layer
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, -1)
        x = self.conv(x)
        out = self.fc(x.view(x.shape[0], -1))
        return out