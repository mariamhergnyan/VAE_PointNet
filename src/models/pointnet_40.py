from torch.utils.data import DataLoader
import torch.utils.data as torch_data


from os.path import join
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim

from tqdm import tqdm
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from IPython import display as ipython_display
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D

# %matplotlib inline


class TNet(nn.Module):
    def __init__(self, dim, num_points=1024):
        super().__init__()
        self.dim = dim
        self.conv1 = torch.nn.Conv1d(dim, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)


        self.fc1 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, dim*3)
        self.relu = nn.ReLU()



        self.max_pool = nn.MaxPool1d(kernel_size=num_points)



    def forward(self, x):
        bs = x.size()[0]

        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.max_pool(x)
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)



        # initialize identity matrix
        iden = torch.eye(self.dim, requires_grad=True).repeat(bs, 1, 1)
        if x.is_cuda:
            iden = iden.cuda()

        x = x.view(-1, self.dim, self.dim) + iden


        return x
class PointNet(nn.Module):
    def __init__(self, num_classes=40, num_points=1024, use_dropout = True):
        super().__init__()
        self.tnet = TNet(3)
        self.use_dropout = use_dropout
        self.num_classes = num_classes

        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.bn1 = torch.nn.BatchNorm1d(64)
        if self.use_dropout:
          self.dropout1d = nn.Dropout(p=0.1)

        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.bn2 = torch.nn.BatchNorm1d(128)
        if self.use_dropout:
          self.dropout2d = nn.Dropout(p=0.2)

        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        self.bn3 = torch.nn.BatchNorm1d(256)
        if self.use_dropout:
          self.dropout3d = nn.Dropout(p=0.3)


        self.conv4 = torch.nn.Conv1d(256, 512, 1)
        self.bn4 = torch.nn.BatchNorm1d(512)
        if self.use_dropout:
          self.dropout4d = nn.Dropout(p=0.3)

        self.conv5 = torch.nn.Conv1d(512, 1024, 1)
        self.bn5 = torch.nn.BatchNorm1d(1024)
        if self.use_dropout:
          self.dropout5d = nn.Dropout(p=0.3)

        self.max_pool = nn.MaxPool1d(kernel_size=num_points)



        self.general_part = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.num_classes),
            nn.Softmax()
        )



    def forward(self, x):
        trans = self.tnet(x)
        x = torch.bmm(x, trans)
        bs = x.size()[0]
        x = x.transpose(2, 1)

        x = F.relu(self.bn1(self.conv1(x)))
        if self.use_dropout:
          x = self.dropout1d(x)

        x = F.relu(self.bn2(self.conv2(x)))
        if self.use_dropout:
          x = self.dropout2d(x)

        x = F.relu(self.bn3(self.conv3(x)))
        if self.use_dropout:
          x = self.dropout1d(x)

        x = F.relu(self.bn4(self.conv4(x)))
        if self.use_dropout:
          x = self.dropout4d(x)

        x = F.relu(self.bn5(self.conv5(x)))
        if self.use_dropout:
          x = self.dropout5d(x)

        # x, _ = torch.max(x, 2)

        #x = self.max_pool(x).view(bs, -1)
        x = self.max_pool(x)
        x = x.view(-1, 1024)

        x = self.general_part(x)

        return x

