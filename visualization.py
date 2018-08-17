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

outputs_all = scipy.io.loadmat('result/prediction_on_test_dataset.mat', mdict=None, appendmat=True)['outputs_all']
predicted_all = scipy.io.loadmat('result/prediction_on_test_dataset.mat', mdict=None, appendmat=True)['predicted_all'].squeeze()
labels_all = scipy.io.loadmat('result/prediction_on_test_dataset.mat', mdict=None, appendmat=True)['labels_all'].squeeze()
paths_all = scipy.io.loadmat('result/prediction_on_test_dataset.mat', mdict=None, appendmat=True)['paths_all']

# to probability
# outputs_all = F.softmax(torch.from_numpy(outputs_all), dim=1)


# visualization
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


# for overlap visualization(probability map)
result_4images = np.zeros((h, w, 4))
for i in range(predicted_all.size):
    prediction = predicted_all[i]
    path = paths_all[i]
    sub_image_row = int(path.split('\\')[-1].split('.')[0].split('_')[2])
    sub_image_column = int(path.split('\\')[-1].split('.')[0].split('_')[3])
    range_row_lower = sub_image_row * stride
    range_column_lower = sub_image_column * stride
    range_row_upper = range_row_lower + sub_image_size
    range_column_upper = range_column_lower + sub_image_size
    # result_4images[range_row_lower:range_row_upper, range_column_lower:range_column_upper, prediction] += 1
    result_4images[range_row_lower:range_row_upper, range_column_lower:range_column_upper, :] += outputs_all[i, :]

result_4images = result_4images / result_4images.max()
result_4images *= 255
result_4images = result_4images.astype(int)

bg = result_4images[:, :, 0]
handwriting = result_4images[:, :, 1]
ruler = result_4images[:, :, 2]
crack = result_4images[:, :, 3]
cv2.imwrite('result/'+ 'bg.jpg', bg)
cv2.imwrite('result/'+ 'hw.jpg', handwriting)
cv2.imwrite('result/'+ 'ru.jpg', ruler)
cv2.imwrite('result/'+ 'cr.jpg', crack)

plt.figure()

plt.subplot(2,2,1)
plt.title('background')
plt.imshow(bg)

plt.subplot(2,2,2)
plt.title('handwriting')
plt.imshow(handwriting)

plt.subplot(2,2,3)
plt.title('ruler')
plt.imshow(ruler)

plt.subplot(2,2,4)
plt.title('crack')
plt.imshow(crack)

plt.savefig('result/' + 'result.jpg')
plt.show()





# # for none overlap visualization
# result_4images = np.zeros((h, w, 4*c), dtype=int)
#
# # iterate over data
# index = 0
# for images, labels, paths in test_loader:
#     # print(len(paths))
#     for i in range(len(paths)):
#         path = paths[i]
#         print(paths)
#         path = path.split('/')[-1].split('.')[0].split('_')
#         image_name = path[0] + '_' + path[1]
#         row, column = int(path[2]), int(path[3])
#         label = labels[i]
#         predicted = predicted_all[index]
#         image = images[i]
#         image = image.numpy()
#         image = np.transpose(image, (1, 2, 0))
#         image = 255 * (image / 2 + 0.5)
#         image = image.astype(int)
#
#         result_4images[row * sub_image_size : (row+1) * sub_image_size, \
#         column * sub_image_size : (column+1) * sub_image_size, \
#         3 * predicted : 3 * predicted + 3] \
#             = image[:, :, :]
#
#         index += 1
#
# bg = result_4images[:, :, 0:3]
# handwriting = result_4images[:, :, 3:6]
# ruler = result_4images[:, :, 6:9]
# crack = result_4images[:, :, 9:12]
#
# plt.figure()
#
# plt.subplot(2,2,1)
# plt.title('background')
# plt.imshow(bg)
#
# plt.subplot(2,2,2)
# plt.title('handwriting')
# plt.imshow(handwriting)
#
# plt.subplot(2,2,3)
# plt.title('ruler')
# plt.imshow(ruler)
#
# plt.subplot(2,2,4)
# plt.title('crack')
# plt.imshow(crack)
#
# plt.savefig('result/' + 'result.jpg')
# plt.show()
# # for none overlap visualization