# APESS2018 Steel Girder Crack Identification
This code is for APESS2018 Steel Girder Crack Identification, it can also be used for other classification tasks.

## Requirements
* numpy
* scipy
* matplotlib
* opencv-python
*
## Usage
* First put raw images to `'image'` folder, put labels to `'label'` folder, please.
* To generate dataset for training and test, please check `'pre_processing.py'`. Training images will be put into `'sub_image_train'` folder, test images will be put into `'sub_image_test'` folder.
* To train, just run `'train.py'`.
* To test, run `'test.py'`, if you want to use parameters from other epochs, please change the model name you want to load.
* To visualize your test result, run `'visualization.py'`.

## References
* Lin, Min, Qiang Chen, and Shuicheng Yan. "Network in network." arXiv preprint arXiv:1312.4400 (2013).
* Dai, Jifeng, et al. "Deformable convolutional networks." CoRR, abs/1703.06211 1.2 (2017): 3.

