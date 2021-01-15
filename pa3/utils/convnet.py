#!/usr/bin/env -S python3 -u

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        conv_in = 3
        conv_h1 = 32
        conv_h2 = 128
        conv_out = 256
        fc_in = conv_out * 4 * 4
        fc_h1 = 1024
        fc_h2 = 76
        fc_out = 10
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(conv_in, conv_h1, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(conv_h1, conv_h2, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(conv_h2, conv_out, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(fc_in, fc_h1),
            nn.BatchNorm1d(fc_h1),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(fc_h1, fc_h2),
            nn.BatchNorm1d(fc_h2),
            nn.ReLU(),
        )
        self.fc3 = nn.Linear(fc_h2, fc_out)

    def forward(self, x):
        z = self.conv1(x)
        z = self.conv2(z)
        z = self.conv3(z)
        z = z.view(z.size()[0], -1)
        z = self.fc1(z)
        z = self.fc2(z)
        z = self.fc3(z)
        z = F.log_softmax(z, dim=1)
        return z


# Author: Blurgy <gy@blurgy.xyz>
# Date:   Jan 15 2021, 14:31 [CST]
