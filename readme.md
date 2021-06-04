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
https://pan.baidu.com/s/1bvuWyIagBcFzNTp4l1eImA ExtractCode:beqg
### 3.2 WorldView-2
https://pan.baidu.com/s/1Gfe6-SGT9AKHD6NzAZyJzA ExtractCode:x0wx
### 3.2 IKONOS
https://pan.baidu.com/s/11rJcRgW0n0OSimzuokH-6Q ExtractCode:4ky4
### 3.4 Perpercess Data
1. You should use the code to clip the image by './data/clip_dataset.py'
2. To train, you should split the dataset by random using the code './data/split_dataset.py
3. Use the code from './MATLAB/custum/degrade_dataset.m' to create the simulate dataset
   

## 4 Pretrained Models

You should put the downloaded pretrained models under one satellite dataset model directory and the related method directory, such as 'QuickBird/checkpoints_dir/deep_learning_method/supervised'.

## 5 Test or Use To Pan-Sharpen
Before testing or using, you must use the train/_test.py to get the setting of the parameters.
As following:
```python
python _test.py -m ours -ms F16B2 -d D:\\实验\\影像融合\\IKONOS\\train_dataset -td D:\\实验\\影像融合\\IKONOS\\test_dataset -sd D:\\实验\\影像融合\\Deep-Learning-PanSharpening\\results\\IKONOS\\result -cd D:\\实验\\影像融合\\Deep-Learning-PanSharpening\\checkpoints\\IKONOS\\checkpoints_dir
```
The detail of the setting can be seen by using --help. After running the file '_test.py', you can get two results in '/your_save_dir/deep_learning_method/supervised/full' and '/your_save_dir/deep_learning_method/supervised/reduce'.

## 6 Evalute
Use the './MATLAB/custum/eval_DL_method.m' to evalute the result

## 7 Train
If training by yourself, firstly you must set the parameters as following:
```python
python train.py -m ours -ms F16B2 -d D:\\实验\\影像融合\\IKONOS\\train_dataset -td D:\\实验\\影像融合\\IKONOS\\test_dataset -sd D:\\实验\\影像融合\\Deep-Learning-PanSharpening\\results\\IKONOS\\result -cd D:\\实验\\影像融合\\Deep-Learning-PanSharpening\\checkpoints\\IKONOS\\checkpoints_dir
```
The detail of the setting can be seen by using --help. After running the file 'train.py', you can get a converge model parameter.

## 8 AcKnowledgement
This code depository is refered to https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix , 
and https://github.com/zk31601102/ResidualDenseNetWork-CycleGAN-SuperResolution

## 9 Cite
to do. 