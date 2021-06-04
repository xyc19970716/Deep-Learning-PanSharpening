import os
import os.path
import random
import numpy as np
import cv2
import torch
import torch.utils.data as data
import config as cfg
import glob
from PIL import Image
import gdal
from torchvision import transforms
from tqdm import tqdm

def gdal_read(input_file):
    # 读取多通道
    # the output format is (h,w,c)
    dataset = gdal.Open(input_file)
    rows = dataset.RasterYSize
    cols = dataset.RasterXSize

    couts = dataset.RasterCount
    data = dataset.ReadAsArray().astype(np.float32) # in order to fit the torch.fromnumpy
  
    if len(data.shape) == 2: # this is one band
        return data.reshape(1, data.shape[0], data.shape[1])
    else:
        return data



def transform(LR, HR, FR, args):
    # this implmented fits the format of c,h,w

    # 随机裁剪
    c, h, w = LR.shape
    size = args.img_size
    location_x = random.randint(0, h-size)
    location_y = random.randint(0, w-size)
    FR = FR[:, location_x*args.scale:(location_x+size)*args.scale, location_y*args.scale:(location_y+size)*args.scale]
    HR = HR[:, location_x*args.scale:(location_x+size)*args.scale, location_y*args.scale:(location_y+size)*args.scale]
    LR = LR[:, location_x:location_x+size, location_y:location_y+size] 

    if random.random() < 0.5:
        # 垂直翻转
        LR = LR[:,::-1,:]
        HR = HR[:,::-1,:]
        FR = FR[:,::-1,:]
    if random.random() < 0.5:
        # 水平翻转
        LR = LR[:,:,::-1]
        HR = HR[:,:,::-1]
        FR = FR[:,:,::-1]
    if random.random() < 0.5:
        # 顺时针旋转90度
        LR = LR.swapaxes(-2,-1)[:,:,::-1]
        HR = HR.swapaxes(-2,-1)[:,:,::-1]
        FR = FR.swapaxes(-2,-1)[:,:,::-1]
    if random.random() < 0.5:
        # 顺时针旋转180度
        LR = LR[:,::-1,:]
        HR = HR[:,::-1,:]
        FR = FR[:,::-1,:]
    if random.random() < 0.5:
        # 顺时针旋转270度
        LR = LR.swapaxes(-2,-1)
        HR = HR.swapaxes(-2,-1)
        FR = FR.swapaxes(-2,-1)
    
    
  
    
    LR, HR, FR = LR.copy(), HR.copy(), FR.copy()
    return LR, HR, FR

class PsDataset(data.Dataset):
    def __init__(self, args, apath, isAug=True, isUnlabel=False):
        self.isAug = isAug
        self.isUnlabel = isUnlabel
        self.scale = args.scale
        # apath = cfg.dataDir
        self.args = args
      
        dirHR = 'HR'
        dirLR = 'LR'
        dirFR = 'FR'
            
        self.dirIn = os.path.join(apath, dirLR)
        self.dirTar = os.path.join(apath, dirHR)
        self.dirFR = os.path.join(apath, dirFR)
        
      
        self.LRList = glob.glob(os.path.join(self.dirIn, '*tif'))
        self.HRList = glob.glob(os.path.join(self.dirTar, '*.tif'))
        self.FRList = glob.glob(os.path.join(self.dirFR, '*.tif'))
        # 随机打乱数据集
        random.seed(args.seed)
        random.shuffle(self.LRList)
        random.seed(args.seed)
        random.shuffle(self.HRList)
        random.seed(args.seed)
        random.shuffle(self.FRList)
      
        self.len = len(self.LRList)
        self.transform = transform
    def __getitem__(self, idx):
        
        LR = gdal_read(self.LRList[idx])
        HR = gdal_read(self.HRList[idx])
        FR = gdal_read(self.FRList[idx])
        lr, hr, fr = LR, HR, FR
        if self.isAug:
            
            lr, hr, fr = self.transform(LR, HR, FR, self.args)
       
   
        lr = torch.Tensor(lr).float() / self.args.data_range
        hr = torch.Tensor(hr).float() / self.args.data_range
        fr = torch.Tensor(fr).float() / self.args.data_range
        return lr, hr, fr
        
    def __len__(self):
        return self.len


def Get_DataSet(dataset, length):
    size = len(length)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    if size == 1:
        flag = int(length[0] * dataset_size)
        return data.Subset(dataset, indices[:flag])
    elif size == 2:
        flag = int(length[0] * dataset_size)
        return data.Subset(dataset, indices[:flag]), data.Subset(dataset, indices[flag:])
    elif size == 3:
        flag1 = int(length[0] * dataset_size)
        flag2 = int(length[1] * dataset_size)
        return data.Subset(dataset, indices[:flag1]), data.Subset(dataset, indices[flag1:flag2]), data.Subset(dataset, indices[flag2:])

