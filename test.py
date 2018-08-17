from __future__ import print_function

import scipy
from scipy import io
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms

import paras


# GPU
USE_GPU = paras.USE_GPU


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


# # data load
# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# test_set = torchvision.datasets.ImageFolder(paras.test_image, transform=transform)
# test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

# data load
class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
test_set = ImageFolderWithPaths(paras.test_image, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)


# test
'''change this if you want to use paras of other epochs'''
net = torch.load('model/model_01.pkl', map_location='cpu')
# GPU
if USE_GPU:
    '''change this if you want to use paras of other epochs'''
    net = torch.load('model/model_01.pkl')

classes = ['0_background', '1_handwriting', '2_ruler', '3_crack']
confusion_matrix = np.zeros((4, 4), dtype=int)
outputs_all = torch.tensor([])
predicted_all = torch.tensor([]).int()
labels_all = torch.tensor([]).int()
paths_all = []
# GPU
if USE_GPU:
    predicted_all = predicted_all.cuda()
    labels_all = labels_all.cuda()
    outputs_all = outputs_all.cuda()
with torch.no_grad():
    for i, data in enumerate(test_loader, 0):
        images, labels, paths = data
        # GPU
        if USE_GPU:
            images = images.cuda()
            labels = labels.cuda()
        outputs = net(images)
        # GPU
        if USE_GPU:
            outputs = outputs.cuda()

        labels = labels.int()

        # from one-hot to scalar label
        predicted = torch.max(outputs, 1)[1].int()
        # GPU
        if USE_GPU:
            predicted = predicted.cuda()

        outputs_all = torch.cat((outputs_all,outputs), 0)
        predicted_all = torch.cat((predicted_all, predicted), 0)
        labels_all = torch.cat((labels_all, labels), 0)
        paths_all += list(paths)

# Confusion matrix
for i in range(labels_all.shape[0]):
    confusion_matrix[predicted_all[i], labels_all[i]] += 1
print('Confusion matrix:\n', confusion_matrix)

# confusion_matrix
mdict = {'confusion_matrix': confusion_matrix}
savename = 'result/confusion_matrix'
scipy.io.savemat(savename, mdict=mdict)
print('Saved to result/confusion_matrix.mat')

# save prediction_on_test_dataset
# GPU
if USE_GPU:
    outputs_all = outputs_all.cpu()
    predicted_all = predicted_all.cpu()
    labels_all = labels_all.cpu()

outputs_all = outputs_all.numpy()
predicted_all = predicted_all.numpy()
labels_all = labels_all.numpy()
mdict = {'outputs_all': outputs_all,
        'predicted_all': predicted_all.squeeze(),
         'labels_all': labels_all.squeeze(),
         'paths_all': paths_all}
savename = 'result/prediction_on_test_dataset'
scipy.io.savemat(savename, mdict=mdict)
print('Saved to result/prediction_on_test_dataset.mat')
