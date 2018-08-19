# APESS2018 Steel Girder Crack Identification
This code is for APESS2018 Steel Girder Crack Identification, it can also be used for other classification tasks. Due to that the parameter quantity is quite few in this network, it can run very fast on both training and test procedures.
![Network architecture](https://github.com/Hufangqiao/APESS2018_Steel_Girder_Crack_Identification/blob/master/network%20architecture.png)

## Requirements
* python 3.6
* numpy 1.14.3
* scipy 1.1.0
* matplotlib 2.2.2
* opencv-python 3.4.2
* pytorch 0.4.1

## Usage
* First put raw images to `'image'` folder, put labels to `'label'` folder, please.
* To generate dataset for training and test, please check `'pre_processing.py'`. Training images will be put into `'sub_image_train'` folder, test images will be put into `'sub_image_test'` folder.
* To train, just run `'train.py'`.
* To test, run `'test.py'`, if you want to use parameters from other epochs, please change the model name you want to load.
* To visualize your test result, run `'visualization.py'`.

## Result
![Result](https://github.com/Hufangqiao/APESS2018_Steel_Girder_Crack_Identification/blob/master/result.jpg)

## P.S.
* If you find this code useful, please consider to cite my next paper :) :)

## References
* Lin, Min, Qiang Chen, and Shuicheng Yan. "Network in network." arXiv preprint arXiv:1312.4400 (2013).
* Dai, Jifeng, et al. "Deformable convolutional networks." CoRR, abs/1703.06211 1.2 (2017): 3.

© 2018 Center of Structural Monitoring and Control, Harbin Institute of Technology
