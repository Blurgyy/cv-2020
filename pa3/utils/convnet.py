#!/usr/bin/env -S python3 -u

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 128, 3, 1, 1),
            nn.ReLU(),
        )
        self.pooling1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
        )
        self.pooling2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Sequential(
            nn.Linear(256 * 8 * 8, 8192),
            nn.BatchNorm1d(8192),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(8192, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        z = self.conv1(x)
        z = self.conv2(z)
        z = self.pooling1(z)
        z = self.conv3(z)
        z = self.pooling2(z)
        z = z.view(z.size()[0], -1)
        z = self.fc1(z)
        z = self.fc2(z)
        z = self.fc3(z)
        z = F.log_softmax(z, dim=1)
        return z


# Author: Blurgy <gy@blurgy.xyz>
# Date:   Jan 15 2021, 14:31 [CST]
