# encoding：utf-8
import numpy as np
import cv2
import scipy
from scipy import io
import glob
import os

import paras


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)


# def func1(raw_image_idx, pix_label_idx):
#     # raw_image = cv2.imread('raw_image/crack_15.jpg')
#     # pix_label = cv2.imread('label/crack_15.png')
#
#     raw_image = cv2.imread(raw_image_idx)
#
#     # # BGR 2 RGB
#     # temp1 = raw_image[:, :, 0]
#     # temp2 = raw_image[:, :, 2]
#     # raw_image[:, :, 0] = temp2
#     # raw_image[:, :, 2] = temp1
#
#     pix_label = cv2.imread(pix_label_idx)
#
#     h, w, c = raw_image.shape
#     sub_image_size = paras.sub_image_size
#     stride = paras.stride
#     rows = int((h-sub_image_size)/stride + 1)
#     columns = int((w-sub_image_size)/stride + 1)
#     class_label_one_hot = np.zeros((rows * columns, 4), dtype=int)
#     class_label = np.zeros((rows * columns, 1), dtype=int)
#     sub_image_array = np.zeros((rows * columns, sub_image_size, sub_image_size, 3), dtype=int)
#
#
#     # class 1
#     blue = [[224, 224, 0], [255, 255, 32]]
#     # class 2
#     yellow = [[0, 224, 224], [32, 255, 255]]
#     # class 3
#     red = [[0, 0, 224], [32, 32, 255]]
#
#
#     lower_blue = np.array(blue[0], dtype="uint8")
#     upper_blue = np.array(blue[1], dtype="uint8")
#     mask1 = cv2.inRange(pix_label, lower_blue, upper_blue)
#
#     lower_yellow = np.array(yellow[0], dtype="uint8")
#     upper_yellow = np.array(yellow[1], dtype="uint8")
#     mask2 = cv2.inRange(pix_label, lower_yellow, upper_yellow)
#
#     lower_red = np.array(red[0], dtype="uint8")
#     upper_red = np.array(red[1], dtype="uint8")
#     mask3 = cv2.inRange(pix_label, lower_red, upper_red)
#
#
#     for column in np.arange(0, columns, 1):
#         for row in np.arange(0, rows, 1):
#             sub_image = raw_image[row*stride:row*stride+sub_image_size, column*stride:column*stride+sub_image_size, :]
#
#             sub_image_pix_label1 = mask1[row * stride:row * stride + sub_image_size,
#                                    column * stride:column * stride + sub_image_size]
#             sub_image_pix_label2 = mask2[row * stride:row * stride + sub_image_size,
#                                    column * stride:column * stride + sub_image_size]
#             sub_image_pix_label3 = mask3[row * stride:row * stride + sub_image_size,
#                                    column * stride:column * stride + sub_image_size]
#
#             if sub_image_pix_label3.sum()>64*255:
#                 sub_image_class_label = 3
#             elif sub_image_pix_label2.sum()>256*255:
#                 sub_image_class_label = 2
#             elif sub_image_pix_label1.sum()>256*255:
#                 sub_image_class_label = 1
#             else:
#                 sub_image_class_label = 0
#
#             # assemble
#             # one hot class label
#             class_label_one_hot[column * rows + row, sub_image_class_label] = 1
#
#             # class label
#             class_label[column * rows + row, :] = sub_image_class_label
#
#
#             sub_image_array[column*rows+row, :, :, :] = sub_image
#
#
#     mdict={'sub_image_array': sub_image_array,
#            'class_label': class_label_one_hot,}
#
#     savename = 'mat/' + os.path.basename(raw_image_idx).split('.')[0]
#
#     scipy.io.savemat(savename, mdict=mdict)


def generate_train_data(raw_image_idx, pix_label_idx):

    classes = ['0_background', '1_handwriting', '2_ruler', '3_crack']

    raw_image = cv2.imread(raw_image_idx)
    pix_label = cv2.imread(pix_label_idx)

    h, w, c = raw_image.shape
    sub_image_size = paras.sub_image_size

    # class 1
    blue = [[224, 224, 0], [255, 255, 32]]
    # class 2
    yellow = [[0, 224, 224], [32, 255, 255]]
    # class 3
    red = [[0, 0, 224], [32, 32, 255]]

    lower_blue = np.array(blue[0], dtype="uint8")
    upper_blue = np.array(blue[1], dtype="uint8")
    mask1 = cv2.inRange(pix_label, lower_blue, upper_blue)

    lower_yellow = np.array(yellow[0], dtype="uint8")
    upper_yellow = np.array(yellow[1], dtype="uint8")
    mask2 = cv2.inRange(pix_label, lower_yellow, upper_yellow)

    lower_red = np.array(red[0], dtype="uint8")
    upper_red = np.array(red[1], dtype="uint8")
    mask3 = cv2.inRange(pix_label, lower_red, upper_red)


    # for background
    stride = paras.stride_bg
    rows = int((h-sub_image_size)/stride + 1)
    columns = int((w-sub_image_size)/stride + 1)

    for column in np.arange(0, columns, 1):
        for row in np.arange(0, rows, 1):
            sub_image = raw_image[row*stride:row*stride+sub_image_size, column*stride:column*stride+sub_image_size, :]

            sub_image_pix_label1 = mask1[row * stride:row * stride + sub_image_size,
                                   column * stride:column * stride + sub_image_size]
            sub_image_pix_label2 = mask2[row * stride:row * stride + sub_image_size,
                                   column * stride:column * stride + sub_image_size]
            sub_image_pix_label3 = mask3[row * stride:row * stride + sub_image_size,
                                   column * stride:column * stride + sub_image_size]

            if sub_image_pix_label3.sum()>256*255:
                sub_image_class_label = 3
            elif sub_image_pix_label2.sum()>1024*255:
                sub_image_class_label = 2
            elif sub_image_pix_label1.sum()>1024*255:
                sub_image_class_label = 1
            else:
                sub_image_class_label = 0

            if sub_image_class_label==0:
                basename = os.path.basename(raw_image_idx).split('.')[0]
                savename = paras.train_image + '/' + classes[sub_image_class_label]\
                           + '/' + basename + '_' + str(row).zfill(2) + '_' \
                           + str(column).zfill(2) + '.jpg'

                # save subimage
                cv2.imwrite(savename, sub_image)


    # for handwriting
    stride = paras.stride_hw
    rows = int((h-sub_image_size)/stride + 1)
    columns = int((w-sub_image_size)/stride + 1)

    for column in np.arange(0, columns, 1):
        for row in np.arange(0, rows, 1):
            sub_image = raw_image[row*stride:row*stride+sub_image_size, column*stride:column*stride+sub_image_size, :]

            sub_image_pix_label1 = mask1[row * stride:row * stride + sub_image_size,
                                   column * stride:column * stride + sub_image_size]
            sub_image_pix_label2 = mask2[row * stride:row * stride + sub_image_size,
                                   column * stride:column * stride + sub_image_size]
            sub_image_pix_label3 = mask3[row * stride:row * stride + sub_image_size,
                                   column * stride:column * stride + sub_image_size]

            if sub_image_pix_label3.sum()>256*255:
                sub_image_class_label = 3
            elif sub_image_pix_label2.sum()>1024*255:
                sub_image_class_label = 2
            elif sub_image_pix_label1.sum()>1024*255:
                sub_image_class_label = 1
            else:
                sub_image_class_label = 0

            if sub_image_class_label==1:
                basename = os.path.basename(raw_image_idx).split('.')[0]
                savename = paras.train_image + '/' + classes[sub_image_class_label]\
                           + '/' + basename + '_' + str(row).zfill(2) + '_' \
                           + str(column).zfill(2) + '.jpg'

                # save subimage
                cv2.imwrite(savename, sub_image)


    # for ruler
    stride = paras.stride_ruler
    rows = int((h-sub_image_size)/stride + 1)
    columns = int((w-sub_image_size)/stride + 1)

    for column in np.arange(0, columns, 1):
        for row in np.arange(0, rows, 1):
            sub_image = raw_image[row*stride:row*stride+sub_image_size, column*stride:column*stride+sub_image_size, :]

            sub_image_pix_label1 = mask1[row * stride:row * stride + sub_image_size,
                                   column * stride:column * stride + sub_image_size]
            sub_image_pix_label2 = mask2[row * stride:row * stride + sub_image_size,
                                   column * stride:column * stride + sub_image_size]
            sub_image_pix_label3 = mask3[row * stride:row * stride + sub_image_size,
                                   column * stride:column * stride + sub_image_size]

            if sub_image_pix_label3.sum()>256*255:
                sub_image_class_label = 3
            elif sub_image_pix_label2.sum()>1024*255:
                sub_image_class_label = 2
            elif sub_image_pix_label1.sum()>1024*255:
                sub_image_class_label = 1
            else:
                sub_image_class_label = 0

            if sub_image_class_label==2:
                basename = os.path.basename(raw_image_idx).split('.')[0]
                savename = paras.train_image + '/' + classes[sub_image_class_label]\
                           + '/' + basename + '_' + str(row).zfill(2) + '_' \
                           + str(column).zfill(2) + '.jpg'

                # save subimage
                cv2.imwrite(savename, sub_image)


    # for crack
    stride = paras.stride_crack
    rows = int((h-sub_image_size)/stride + 1)
    columns = int((w-sub_image_size)/stride + 1)

    for column in np.arange(0, columns, 1):
        for row in np.arange(0, rows, 1):
            sub_image = raw_image[row*stride:row*stride+sub_image_size, column*stride:column*stride+sub_image_size, :]

            sub_image_pix_label1 = mask1[row * stride:row * stride + sub_image_size,
                                   column * stride:column * stride + sub_image_size]
            sub_image_pix_label2 = mask2[row * stride:row * stride + sub_image_size,
                                   column * stride:column * stride + sub_image_size]
            sub_image_pix_label3 = mask3[row * stride:row * stride + sub_image_size,
                                   column * stride:column * stride + sub_image_size]

            if sub_image_pix_label3.sum()>256*255:
                sub_image_class_label = 3
            elif sub_image_pix_label2.sum()>1024*255:
                sub_image_class_label = 2
            elif sub_image_pix_label1.sum()>1024*255:
                sub_image_class_label = 1
            else:
                sub_image_class_label = 0

            if sub_image_class_label==3:
                basename = os.path.basename(raw_image_idx).split('.')[0]
                savename = paras.train_image + '/' + classes[sub_image_class_label]\
                           + '/' + basename + '_' + str(row).zfill(2) + '_' \
                           + str(column).zfill(2) + '.jpg'

                # save subimage
                cv2.imwrite(savename, sub_image)


def generate_test_data(raw_image_idx, pix_label_idx):

    classes = ['0_background', '1_handwriting', '2_ruler', '3_crack']

    raw_image = cv2.imread(raw_image_idx)
    pix_label = cv2.imread(pix_label_idx)

    h, w, c = raw_image.shape
    sub_image_size = paras.sub_image_size
    stride = paras.stride
    rows = int((h-sub_image_size)/stride + 1)
    columns = int((w-sub_image_size)/stride + 1)

    # class 1
    blue = [[224, 224, 0], [255, 255, 32]]
    # class 2
    yellow = [[0, 224, 224], [32, 255, 255]]
    # class 3
    red = [[0, 0, 224], [32, 32, 255]]

    lower_blue = np.array(blue[0], dtype="uint8")
    upper_blue = np.array(blue[1], dtype="uint8")
    mask1 = cv2.inRange(pix_label, lower_blue, upper_blue)

    lower_yellow = np.array(yellow[0], dtype="uint8")
    upper_yellow = np.array(yellow[1], dtype="uint8")
    mask2 = cv2.inRange(pix_label, lower_yellow, upper_yellow)

    lower_red = np.array(red[0], dtype="uint8")
    upper_red = np.array(red[1], dtype="uint8")
    mask3 = cv2.inRange(pix_label, lower_red, upper_red)

    for column in np.arange(0, columns, 1):
        for row in np.arange(0, rows, 1):
            sub_image = raw_image[row*stride:row*stride+sub_image_size, column*stride:column*stride+sub_image_size, :]

            sub_image_pix_label1 = mask1[row * stride:row * stride + sub_image_size,
                                   column * stride:column * stride + sub_image_size]
            sub_image_pix_label2 = mask2[row * stride:row * stride + sub_image_size,
                                   column * stride:column * stride + sub_image_size]
            sub_image_pix_label3 = mask3[row * stride:row * stride + sub_image_size,
                                   column * stride:column * stride + sub_image_size]

            if sub_image_pix_label3.sum()>256*255:
                sub_image_class_label = 3
            elif sub_image_pix_label2.sum()>2048*255:
                sub_image_class_label = 2
            elif sub_image_pix_label1.sum()>2048*255:
                sub_image_class_label = 1
            else:
                sub_image_class_label = 0

            basename = os.path.basename(raw_image_idx).split('.')[0]
            savename = paras.test_image + '/' + classes[sub_image_class_label]\
                       + '/' + basename + '_' + str(row).zfill(2) + '_' \
                       + str(column).zfill(2) + '.jpg'

            # save subimage
            cv2.imwrite(savename, sub_image)


if __name__ == '__main__':

    raw_image_list = glob.glob('image/*.jpg')
    pix_label_list = glob.glob('label/*.png')
    raw_image_list.sort()
    pix_label_list.sort()

    classes = ['0_background', '1_handwriting', '2_ruler', '3_crack']
    sub_image_size = paras.sub_image_size

    # generate train data
    for i in classes:
        file_name = paras.train_image + '/' + str(i)
        mkdir(file_name)
    for k in np.arange(len(raw_image_list)):
        raw_image_idx = raw_image_list[k]
        pix_label_idx = pix_label_list[k]
        generate_train_data(raw_image_idx, pix_label_idx)

    # generate test data
    for i in classes:
        file_name = paras.test_image + '/' + str(i)
        mkdir(file_name)
    # change 'test_image' in 'paras.py' if you want to test other image
    idx = paras.test_image.split('/')[1]
    raw_image_idx = 'image/' + idx + '.jpg'
    pix_label_idx = 'label/' + idx + '.png'
    generate_test_data(raw_image_idx, pix_label_idx)

    mkdir('model')
    mkdir('result')
