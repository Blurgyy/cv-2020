#!/usr/bin/env -S python3 -u

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):

    def __init__(self):
        conv_input = 1
        conv_hidden1 = 6
        conv_output = 16
        fc_input = 6 * 6 * 16
        fc_h1 = 120
        fc_h2 = 84
        fc_output = 10

        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=conv_input,
                out_channels=conv_hidden1,
                kernel_size=3,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=conv_hidden1,
                out_channels=conv_output,
                kernel_size=3,
                stride=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=fc_input, out_features=fc_h1),
            nn.BatchNorm1d(fc_h1),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=fc_h1, out_features=fc_h2),
            nn.BatchNorm1d(fc_h2),
            nn.ReLU(),
        )
        self.fc3 = nn.Linear(in_features=fc_h2, out_features=fc_output)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x

    def nfeatures(self, x):
        size = x.size()[1:]
        nfeats = 1
        for s in size:
            nfeats *= s
        return nfeats


def main():
    net = LeNet()
    x = torch.randn(1, 1, 28, 28)
    out = net(x)
    print(out.size())
    print(out)


if __name__ == "__main__":
    main()

# Author: Blurgy <gy@blurgy.xyz>
# Date:   Jan 09 2021, 14:52 [CST]
