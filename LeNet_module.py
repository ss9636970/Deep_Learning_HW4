import random
import numpy as np
import torch.nn as nn
import torch

class LeNet_module(nn.Module):
    def __init__(self):
        super(LeNet_module, self).__init__()
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, stride=2)

        self.conv1 = nn.Sequential(nn.Conv2d(3, 6, 5), self.relu, self.maxpool)
        self.conv2 = nn.Sequential(nn.Conv2d(6, 16, 5), self.relu, self.maxpool)
        self.conv3 = nn.Sequential(nn.Conv2d(16, 26, 5), self.relu, self.maxpool)
        self.conv4 = nn.Sequential(nn.Conv2d(26, 36, 5), self.relu, self.maxpool)
        self.conv5 = nn.Sequential(nn.Conv2d(36, 46, 1), self.relu, self.maxpool)

        self.fc1 = nn.Sequential(nn.Linear(46*2*2, 100), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(100, 50), nn.ReLU())

    def forward(self, X):
        n = X.shape[0]
        h1 = self.conv1(X)
        h2 = self.conv2(h1)
        h3 = self.conv3(h2)
        h4 = self.conv4(h3)
        h5 = self.conv5(h4)
        h5 = h5.reshape(n, -1)
        h6 = self.fc1(h5)
        outputs = self.fc2(h6)
        return outputs