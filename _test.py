import numpy as np
import cv2
import os
import glob
from scipy import signal

# DL-methods
from model.PNN import PNNModel
from model.PanNet import PanNetModel
from model.Ours import OursModel
from model.TFNet import TFNetModel
from model.MSDCNN import MSDCNNModel
from model.SRPPNN import SRPPNNModel


import cupy as cp
from cupy.core.dlpack import toDlpack
from cupy.core.dlpack import fromDlpack
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack

import gdal
from config import data_range
from config import testdataDir
import config as cfg
from tqdm import tqdm
import torch
import time
from thop import profile

def gdal_read(input_file):
    # 读取多通道
    # the output format is (h,w,c)
    dataset = gdal.Open(input_file)
    geo = dataset.GetGeoTransform()
    proj = dataset.GetProjection()
    data = dataset.ReadAsArray() # in order to fit the torch.fromnumpy
    
    
    if len(data.shape) == 2: # this is one band
        return geo, proj, data.reshape(data.shape[0], data.shape[1], 1)
    else:
        return geo, proj, data.transpose((1,2,0)) # in order to use the cv transform

def gdal_write(output_file,array_data, geo, proj):

    #判断栅格数据的数据类型0
    
    if 'int8' in array_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in array_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    h,w,c = array_data.shape
    Driver = gdal.GetDriverByName("Gtiff")
    dataset = Driver.Create(output_file,w,h,c,datatype)
    
    dataset.SetGeoTransform(geo)
    dataset.SetProjection(proj)
    for i in range(c):
        band = dataset.GetRasterBand(i+1)
        band.WriteArray(array_data[:,:,i])


input_data_path = testdataDir
input_ms_path = glob.glob(os.path.join(os.path.join(input_data_path, 'MS'), '*.tif'))
input_pan_path = glob.glob(os.path.join(os.path.join(input_data_path, 'PAN'), '*.tif'))
print(input_ms_path, input_pan_path)
input_ms_down_path = glob.glob(os.path.join(os.path.join(input_data_path, 'LR'), '*.tif'))
input_pan_down_path = glob.glob(os.path.join(os.path.join(input_data_path, 'HR'), '*.tif'))

# 创建存储路径
if cfg.save_result:
    # dl方法

    ours_path = os.path.join(cfg.saveDir, cfg.model+cfg.model_sub+cfg.model_loss)
    

    # dl supervised
    
    supervised_path = os.path.join(ours_path, 'supervised')
    
    
    reduce_path = os.path.join(supervised_path, 'reduce')
    full_path = os.path.join(supervised_path, 'full')
    
   
    if not os.path.exists(ours_path):
        os.makedirs(ours_path)
    

    
    if not os.path.exists(supervised_path):
        os.makedirs(supervised_path)
  

    
    
    if not os.path.exists(reduce_path):
        os.makedirs(reduce_path)
    if not os.path.exists(full_path):
        os.makedirs(full_path)

    
    # dl unsupervised
   
    ours_path = os.path.join(cfg.saveDir, cfg.model)
 
   
    unsupervised_path = os.path.join(ours_path, 'unsupervised')
   

   
    fake_path = os.path.join(unsupervised_path, 'fake')
    unfull_path = os.path.join(unsupervised_path, 'full')
    unreduce_path = os.path.join(unsupervised_path, 'reduce')

    if not os.path.exists(unsupervised_path):
        os.makedirs(unsupervised_path)


    if not os.path.exists(fake_path):
        os.makedirs(fake_path)
    if not os.path.exists(unfull_path):
        os.makedirs(unfull_path)
    if not os.path.exists(unreduce_path):
        os.makedirs(unreduce_path)

# 选择
if cfg.model == 'ours':
    model = OursModel()

elif cfg.model == 'PNN':
    model = PNNModel()

elif cfg.model == 'PanNet':
    model = PanNetModel()
    
elif cfg.model == 'TFNet':
    model = TFNetModel()
    
elif cfg.model == 'MSDCNN':
    model = MSDCNNModel()
   
elif cfg.model == 'SRPPNN':
    model = SRPPNNModel()


model.initialize()

model.load_networks(999)

full_times = []
reduce_times = []

for i, ms_path in tqdm(enumerate(input_ms_path)):
    # print(i)
    step = i
    
    '''loading data'''
    _, _, used_ms = gdal_read(ms_path)
    geo, proj, used_pan = gdal_read(input_pan_path[i])
    _, _, used_ms_down = gdal_read(input_ms_down_path[i])
    geo_down, proj_down, used_pan_down = gdal_read(input_pan_down_path[i])

    '''normalization'''
    used_ms = used_ms / data_range
    used_pan = used_pan / data_range
    used_ms_down = used_ms_down / data_range
    used_pan_down = used_pan_down / data_range

    name = os.path.basename(ms_path)    
    
    if cfg.isUnlabel:
        fused_image = model.predict(used_ms[:, :, :], used_pan[:, :, :])
        if cfg.save_result:
            gdal_write(os.path.join(unfull_path,name), fused_image, geo)
        fused_down_image = model.predict(used_ms_down[:, :, :], used_pan_down[:, :, :])
        if cfg.save_result:
            gdal_write(os.path.join(unreduce_path,name), fused_down_image, geo_down)
    else:
        s_time = time.time()
        
        fused_image = model.predict(used_ms[:, :, :], used_pan[:, :, :])
        e_time = time.time()
        full_times.append(e_time-s_time)

        if cfg.save_result:
            gdal_write(os.path.join(full_path,name), fused_image, geo, proj)

        s_time = time.time()
       
        fused_down_image = model.predict(used_ms_down[:, :, :], used_pan_down[:, :, :])
        e_time = time.time()
        reduce_times.append(e_time-s_time)

        if cfg.save_result:
           gdal_write(os.path.join(reduce_path,name), fused_down_image, geo_down, proj_down)

full_times = np.array(full_times)
reduce_times = np.array(reduce_times)

print('full_time:{}±{}'.format(np.mean(full_times), np.var(full_times)))
print('reduce_time:{}±{}'.format(np.mean(reduce_times), np.var(reduce_times)))
   
    
    
    





