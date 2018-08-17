from __future__ import print_function

import scipy
from scipy import io
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import glob
import os


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms

import paras
from network import Net


# GPU
USE_GPU = paras.USE_GPU

# regularzation
USE_REG = paras.USE_REG


# image parameters
h, w, c = paras.h, paras.w, paras.c
sub_image_size = paras.sub_image_size
stride = paras.stride
rows = int((h - sub_image_size) / stride + 1)
columns = int((w - sub_image_size) / stride + 1)


# hyper-parameters
lr = paras.lr
momentum = paras.momentum
epoch = paras.epoch
batch_size = paras.batch_size

# data load
transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     # transforms.RandomVerticalFlip(),
     transforms.RandomGrayscale(),
     # transforms.RandomAffine((-30, 30)),
     # transforms.RandomRotation((-60, 60)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_set = torchvision.datasets.ImageFolder(paras.train_image, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

# network
net = Net()
if USE_GPU:
    net = net.cuda()

# CrossEntropy
criterion = nn.CrossEntropyLoss()

# SGD
# optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)


# train
loss_all = []
l1 = 0
l2 = 0
for e in range(epoch):

    if (e<3):
        lr = paras.lr
    elif(e>=3 and e<8):
        lr = paras.lr/3
    elif(e>=8 and e<14):
        lr = paras.lr/10
    elif(e>=14):
        lr = paras.lr/50

    # Adam
    optimizer = optim.Adam(net.parameters(), lr=lr)


    # running_loss = 0.0
    for i, data in enumerate(train_loader, 0):

        inputs, labels = data  # zero the parameter gradients
        # GPU
        if USE_GPU:
            inputs = inputs.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()

        # forward
        outputs = net(inputs)
        # GPU
        if USE_GPU:
            outputs = outputs.cuda()

        # for class imbalance CrossEntropy
        labels_one_hot = torch.zeros((labels.shape[0], 4))
        # GPU
        if USE_GPU:
            labels_one_hot = labels_one_hot.cuda()

        for i in range(labels.shape[0]):
            labels_one_hot[i, labels[i]] += 1

        weight = 1 /torch.sum(labels_one_hot, 0).float()
        # GPU
        if USE_GPU:
            weight = weight.cuda()

        criterion.weight = weight
        # loss = criterion(outputs, torch.max(labels, 1)[1]) # for one-hot

        loss = criterion(outputs, labels)
        if USE_GPU:
            loss = loss.cuda()

        if USE_REG:
            lambda_l1 = paras.lambda_l1
            lambda_l2 = paras.lambda_l2
            for p in net.parameters():
                l1 = l1 + p.abs().sum()
                l2 = l2 + (p.abs() * p.abs()).sum()
            loss = loss + lambda_l2 * l2

            print(lambda_l2 * l2)

        # backward + optimization
        loss.backward(retain_graph=True)
        optimizer.step()

        # print statistics
        # running_loss += loss.item()
        running_loss = loss.item()
        loss_all.append(running_loss)


        print('[%d, %5d] loss: %.3f' %(e + 1, i + 1, running_loss))
        # running_loss = 0.0

    # save model
    torch.save(net.state_dict(), 'model/params_' + str(e+1).zfill(2) + '.pkl')
    torch.save(net, 'model/model_' + str(e+1).zfill(2) + '.pkl')
    print('Finished training, model was saved to model/')

# save loss_all
mdict = {'loss_all': loss_all}
savename = 'result/loss_all'
scipy.io.savemat(savename, mdict=mdict)
print('Saved to result/loss_all.mat')

plt.figure()
plt.plot(np.arange(0, len(loss_all)), loss_all)
plt.show()