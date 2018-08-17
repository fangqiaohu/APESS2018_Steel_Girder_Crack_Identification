# GPU
USE_GPU = True

# regularzation
USE_REG = False


# image parameters
h, w, c = 3264, 4928, 3
sub_image_size = 96
stride = 16
stride_bg = 128
stride_hw = 64
stride_ruler = 48
stride_crack = 24
rows = int((h - sub_image_size) / stride + 1)
columns = int((w - sub_image_size) / stride + 1)

# image folder
train_image = 'sub_image_train/'

'''
change this if you want to test other image, do not forget to generate test data again using 'pre_processing.py', 
once you have generated training data, you can add annotations before each line related to generating training data.
'''
test_image = 'sub_image_test/crack_35/'


# hyper-parameters
lr = 3e-4
momentum=0.9
epoch = 20
batch_size = 128
lambda_l1 = 0.0001
lambda_l2 = 0.00002