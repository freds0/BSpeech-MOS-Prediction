import torch.nn as nn
import torch.nn.functional as F
import torch

class EmbeddingFullyConnected(nn.Module):

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