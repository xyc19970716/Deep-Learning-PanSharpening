# Deep Learning PanSharpening 

## 1 Description
This is a deep-learning based pan-sharpening code package, we reimplemented include PNN, MSDCNN, PanNet, TFNet, SRPPNN, and our purposed network DIPNet.

## 2 Environment Setup
This code has been tested on on a personal computer with AMD Ryzen 5 3600 3.6GHz processor, 32-GB RAM, and an NVIDIA RTX2070Super graphic card, Python 3.7.4, Pytorch-1.6.0, CUDA 10.2, cuDNN 8.0.2. 

If you wanted to run our code package in your host, please use the following code in our depository to complete the runtime library.
```python
pip install -r requirements.txt
```

## 3 Dataset
### 3.1 QuickBird 

### 3.2 WorldView-2

### 3.2 IKONOS

### 3.4 Perpercess Data
1. You should use the code to clip the image by './data/clip_dataset.py'
2. To train, you should split the dataset by random using the code './data/split_dataset.py
3. Use the code from './MATLAB/custum/degrade_dataset.m' to create the simulate dataset
   

## 4 Pretrained Models

You should put the downloaded pretrained models under one satellite dataset model directory and the related method directory, such as 'QuickBird/checkpoints_dir/deep_learning_method/supervised'.

## 5 Test or Use To Pan-Sharpen
Before testing or using, you must set the 'config.py' in our main directory. 
As following:
```python
model='PNN' # deep learning pan-sharpening method
save_result = True # whether or not to save result
saveDir = r'D:\实验\影像融合\IKONOS\result' # save directory
checkpoints_dir = r'D:\实验\影像融合\IKONOS\checkpoints_dir' # the dir of pertrained models
```
The detail of the setting can be seen in the 'config.py'. After running the file '_test.py', you can get two results in '/your_save_dir/deep_learning_method/supervised/full' and '/your_save_dir/deep_learning_method/supervised/reduce'.

## 6 Evalute
Use the './MATLAB/custum/eval_DL_method.m' to evalute the result

## 7 Train
If training by yourself, firstly you must set the 'config.py' in our main directory. As following:
```python
model ='PNN' # the model you wanted to train 
dataDir = r'D:\实验\影像融合\IKONOS\train_dataset' # this is a train dataset dir
testdataDir = r'D:\实验\影像融合\IKONOS\test_dataset' # this is a test dataset dir
pan_channel = 1 # pan-chromatic band
mul_channel = 4 # multi-spectral band
```
The detail of the setting can be seen in the 'config.py'. After running the file 'train.py', you can get a converge model parameter.

## 8 AcKnowledgement
This code depository is refered to https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix , 
and https://github.com/zk31601102/ResidualDenseNetWork-CycleGAN-SuperResolution

## 9 Cite
to do. 