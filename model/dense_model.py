import torch.nn as nn
import torch.nn.functional as F
import torch

class dense_model(nn.Module):

    def __init__(self, input_dim=1024):
        super().__init__()
        self.dense1 = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 2048)
        )
        self.dense2 = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1)
        )

    def forward(self, features):
        x = self.dense1(features)
        x = self.dense2(x)
        return x


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.avg_pool2d(x, 2)
        return x


class cnn2d_model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            conv_block(in_channels=1, out_channels=64),
            conv_block(in_channels=64, out_channels=128),
            conv_block(in_channels=128, out_channels=256),
            #conv_block(in_channels=256, out_channels=512),
        )

        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(64*256, 128),
            nn.PReLU(),
            #nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        #print(x.shape)
        x = self.conv(x)
        #print(x.shape)
        x = torch.mean(x, dim=2)
        #print(x.shape)
        #x, _ = torch.max(x, dim=2)
        x = x.view(x.shape[0], -1)
        #print(x.shape)
        x = self.fc(x)
        #print(x.shape)
        return x
