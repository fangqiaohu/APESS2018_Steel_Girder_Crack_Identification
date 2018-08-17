from __future__ import print_function

import matplotlib
matplotlib.use('TkAgg')

import torch
import torch.nn as nn
import torch.nn.functional as F

import paras
from torch_deform_conv.layers import ConvOffset2D

USE_GPU = paras.USE_GPU


class Net(nn.Module):

    # def __init__(self):
    #     super(Net, self).__init__()
    #     self.offset1 = ConvOffset2D(3)
    #     self.conv1 = nn.Conv2d(3, 16, 5)
    #     self.pool1 = nn.MaxPool2d(2, 2)
    #     self.offset2 = ConvOffset2D(16)
    #     self.conv2 = nn.Conv2d(16, 32, 5)
    #     self.pool2 = nn.MaxPool2d(2, 2)
    #     self.fc1 = nn.Linear(21 * 21 * 32, 512)
    #     self.fc2 = nn.Linear(512, 64)
    #     self.fc3 = nn.Linear(64, 4)
    #     self.softmax = nn.Softmax(dim=1)
    #
    # def forward(self, x):
    #     # x = self.offset1(x)
    #     x = self.conv1(x)
    #     x = F.relu(x)
    #     x = self.pool1(x)
    #     # x = self.offset2(x)
    #     x = self.conv2(x)
    #     x = F.relu(x)
    #     x = self.pool2(x)
    #     x = x.view(-1, 21 * 21 * 32)
    #     x = self.fc1(x)
    #     x = F.relu(x)
    #     x = self.fc2(x)
    #     x = F.relu(x)
    #     x = self.fc3(x)
    #     x = self.softmax(x)
    #     return x

    def __init__(self):
        super(Net, self).__init__()

        # branch 1
        self.conv11 = nn.Conv2d(3, 16, 5)
        self.bn11 = nn.BatchNorm2d(16)
        self.pool11 = nn.MaxPool2d(2, 2)
        self.conv12 = nn.Conv2d(16, 32, 5)
        self.bn12 = nn.BatchNorm2d(32)
        self.pool12 = nn.MaxPool2d(2, 2)
        self.conv13 = nn.Conv2d(32, 64, 3)
        self.bn13 = nn.BatchNorm2d(64)
        self.pool13 = nn.MaxPool2d(2, 2)
        self.conv14 = nn.Conv2d(64, 128, 3)
        self.bn14 = nn.BatchNorm2d(128)
        self.gap1 = nn.AvgPool2d(7, 7)

        # branch 2
        self.offset21 = ConvOffset2D(3)
        self.conv21 = nn.Conv2d(3, 8, 5)
        self.bn21 = nn.BatchNorm2d(8)
        self.pool21 = nn.MaxPool2d(2, 2)
        self.offset22 = ConvOffset2D(8)
        self.conv22 = nn.Conv2d(8, 16, 5)
        self.bn22 = nn.BatchNorm2d(16)
        self.pool22 = nn.MaxPool2d(2, 2)
        self.offset23 = ConvOffset2D(16)
        self.conv23 = nn.Conv2d(16, 32, 3)
        self.bn23 = nn.BatchNorm2d(32)
        self.pool23 = nn.MaxPool2d(2, 2)
        self.offset24 = ConvOffset2D(32)
        self.conv24 = nn.Conv2d(32, 64, 3)
        self.bn24 = nn.BatchNorm2d(64)
        self.gap2 = nn.AvgPool2d(7, 7)

        # mainstream
        # self.fc1 = nn.Linear(3 * 3 * 192, 1024)
        # self.fc2 = nn.Linear(1024, 64)
        # self.fc3 = nn.Linear(64, 4)
        self.conv5 = nn.Conv2d(192, 64, 1)
        self.conv6 = nn.Conv2d(64, 16, 1)
        self.conv7 = nn.Conv2d(16, 4, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):

        # branch 1
        x1 = self.conv11(input)
        # x1 = self.bn11(x1)
        x1 = F.relu(x1)
        x1 = self.pool11(x1)

        x1 = self.conv12(x1)
        # x1 = self.bn12(x1)
        x1 = F.relu(x1)
        x1 = self.pool12(x1)

        x1 = self.conv13(x1)
        # x1 = self.bn13(x1)
        x1 = F.relu(x1)
        x1 = self.pool13(x1)

        x1 = self.conv14(x1)
        # x1 = self.bn14(x1)
        x1 = F.relu(x1)

        x1 = self.gap1(x1)

        # branch 2
        x2 = self.offset21(input)
        x2 = self.conv21(x2)
        # x2 = self.bn21(x2)
        x2 = F.relu(x2)
        x2 = self.pool21(x2)

        x2 = self.offset22(x2)
        x2 = self.conv22(x2)
        # x2 = self.bn22(x2)
        x2 = F.relu(x2)
        x2 = self.pool22(x2)

        x2 = self.offset23(x2)
        x2 = self.conv23(x2)
        # x2 = self.bn23(x2)
        x2 = F.relu(x2)
        x2 = self.pool23(x2)

        x2 = self.offset24(x2)
        x2 = self.conv24(x2)
        # x2 = self.bn24(x2)
        x2 = F.relu(x2)

        x2 = self.gap1(x2)

        # mainstream
        x = torch.cat((x1, x2), 1)
        # x = x.view(-1, 3 * 3 * 192)
        # x = self.fc1(x)
        # x = F.relu(x)
        # x = self.fc2(x)
        # x = F.relu(x)
        # x = self.fc3(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.conv7(x)
        x = x.squeeze()
        x = self.softmax(x)
        # print(x.shape)
        return x

    # def __init__(self):
    #     super(Net, self).__init__()
    #     self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
    #     self.bn1 = nn.BatchNorm2d(32)
    #
    #     self.offset2 = ConvOffset2D(32)
    #     self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
    #     self.bn2 = nn.BatchNorm2d(64)
    #
    #     self.offset3 = ConvOffset2D(64)
    #     self.conv3 = nn.Conv2d(64, 32, kernel_size=3)
    #     self.bn3 = nn.BatchNorm2d(32)
    #
    #     self.final_offset = ConvOffset2D(32)
    #     self.final_conv = nn.Conv2d(32, 4, kernel_size=3)
    #
    #     self.fc1 = nn.Linear(10 * 10 * 32, 4)
    #
    #     self.pool = nn.MaxPool2d(2, 2)
    #     self.global_average_pooling = nn.AvgPool2d(8, 8)
    #     self.softmax = nn.Softmax(dim=1)
    #     self.log_softmax = nn.LogSoftmax(dim=1)
    #     self.drop = nn.Dropout2d()
    #
    # def forward(self, x):
    #
    #     # GPU
    #     if USE_GPU:
    #         x = x.cuda()
    #
    #     x = self.conv1(x)
    #     # x = F.prelu(x, 0.1)
    #     x = F.relu(x)
    #     x= self.bn1(x)
    #
    #     # pooling
    #     x = self.pool(x)
    #
    #     x = self.offset2(x)
    #     x = self.conv2(x)
    #     x = F.relu(x)
    #     x = self.bn2(x)
    #
    #     # pooling
    #     x = self.pool(x)
    #
    #     x = self.offset3(x)
    #     x = self.conv3(x)
    #     x = F.relu(x)
    #     x = self.bn3(x)
    #
    #     # pooling
    #     x = self.pool(x)
    #
    #     # # final conv
    #     # x = self.final_offset(x)
    #     # x = self.final_conv(x)
    #     # x = F.relu(x)
    #
    #     # # global_average_pooling
    #     # x = self.global_average_pooling(x)
    #     # x = x.squeeze()
    #
    #     # fc
    #     x = x.view(-1, 10 * 10 * 32)
    #     x = self.fc1(x)
    #
    #     # softmax
    #     x = self.softmax(x)
    #
    #     return x
