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

from tqdm import tqdm
import torch
import time
from thop import profile
import argparse
from mlab.releases import latest_release as matlab

def ERGAS(I1, I2, ratio=4):
    matlab.path(matlab.path(),r'D:\实验\影像融合\其他融合方法\Pansharpening Tool ver 1.3\Quality_Indices')#设置路径
    index = matlab.ERGAS(I1,I2,[ratio])#调用自己定义的m函数就可以了
    return index

def SAM(I1,I2):
    matlab.path(matlab.path(),r'D:\实验\影像融合\其他融合方法\Pansharpening Tool ver 1.3\Quality_Indices')#设置路径
    index = matlab.SAM(I1,I2)#调用自己定义的m函数就可以了
    return index

def SCC(I1,I2):
    matlab.path(matlab.path(),r'D:\实验\影像融合\其他融合方法\Pansharpening Tool ver 1.3\Quality_Indices')#设置路径
    index = matlab.SCC(I1,I2)#调用自己定义的m函数就可以了
    return index

def Q(I1,I2):
    # I1 融合
    # I2 GT
    matlab.path(matlab.path(),r'D:\实验\影像融合\其他融合方法\Pansharpening Tool ver 1.3\Quality_Indices')#设置路径
    index = matlab.q2n(I2,I1, [32], [32])#调用自己定义的m函数就可以了
    return index

def QAVE(I1,I2):
    # I1 融合
    # I2 GT
    matlab.path(matlab.path(),r'D:\实验\影像融合\其他融合方法\Pansharpening Tool ver 1.3\Quality_Indices')#设置路径
    index = matlab.Q(I2,I1, [2048])#调用自己定义的m函数就可以了
    return index

def D_lambda(I1,I2):
    # I1 融合
    # I2 放大MS
    matlab.path(matlab.path(),r'D:\实验\影像融合\其他融合方法\Pansharpening Tool ver 1.3\Quality_Indices')#设置路径
    index = matlab.D_lambda(I1,I2, [32], [1])#调用自己定义的m函数就可以了
    return index

def D_s(I1,I2,I3, args):
    # I1 融合
    # I2 放大MS
    # I3 PAN
    matlab.path(matlab.path(),r'D:\实验\影像融合\其他融合方法\Pansharpening Tool ver 1.3\Quality_Indices')#设置路径
    index = matlab.D_s(I1,I2,I3, str(args.sensor),[args.scale],[32], [1])#调用自己定义的m函数就可以了

    return index

def QNR(D_s, D_lambda):
    alpha = 1
    beta = 1
    return (1-D_lambda)**alpha * (1-D_s)**beta

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


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="ours", help="deep learning pan-sharpening method such as PNN MSDCNN PanNet TFNet SRPPNN and DIPNet")
parser.add_argument("-ms", "--model_sub", type=str, default="", help="for our DIPNet's ablation study, such as SR SR_PLB SR_PHB SR_PLB_PHB default set as ''")
parser.add_argument("-ml", "--model_loss", type=str, default="", help="for our DIPNet's ablation study, such as SSIM or L1 or '', default set as ''")
parser.add_argument("-t", "--isTrain", type=bool, default=True, help="whether or not to train")
parser.add_argument("-g", "--gpu_ids", type=int, default=0, help="")
parser.add_argument("-c", "--continue_train", type=bool, default=False, help="whether or not to continue")
parser.add_argument("-we", "--which_epoch", type=int, default=-1, help="if continue Train ture, set this value")
parser.add_argument("-p", "--print_net_in_detail", type=bool, default=False, help="whether or not to continue")
parser.add_argument("-d", "--dataDir", type=str, default="", help="this is a train dataset dir")
parser.add_argument("-td", "--testdataDir", type=str, default="", help="this is a test dataset dir ")
parser.add_argument("-sr", "--save_result", type=bool, default=True, help="whether or not to save result")
parser.add_argument("-sd", "--saveDir", type=str, default="", help="save directory")
parser.add_argument("-cd", "--checkpoints_dir", type=str, default="", help="the dir of pertrained models or the dir to checkpoint the model parameters during training")
parser.add_argument("-nT", "--nThreads", type=int, default=0, help="use serval threads to load data in pytorch")
parser.add_argument("-bs", "--batchSize", type=int, default=16, help="")
parser.add_argument("-is", "--img_size", type=int, default=32, help="simulate ms size")
parser.add_argument("-sa", "--scale", type=int, default=4, help="scale factor which resizing to pan size")
parser.add_argument("-se", "--seed", type=int, default=19970716, help="random seed")
parser.add_argument("-pf", "--print_freq", type=int, default=5, help="print frequency of log")
parser.add_argument("-sf", "--save_epoch_freq", type=int, default=500, help="print frequency of log")
parser.add_argument("-pc", "--pan_channel", type=int, default=1, help="pan-chromatic band")
parser.add_argument("-mc", "--mul_channel", type=int, default=4, help="multi-spectral band which is based on different satellite")
parser.add_argument("-gm", "--gan_mode", type=str, default="lsgan", help="'lsgan' or 'wgangp' or 'vanilla' this is orginal")
parser.add_argument("-dr", "--data_range", type=int, default=2047, help="radis resolution ")
parser.add_argument("-lp", "--lr_policy", type=str, default='step', help="")
parser.add_argument("-ot", "--optim_type", type=str, default='adam', help="optim_type")
parser.add_argument("-lr", "--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("-b", "--beta", type=float, default=0.9, help="")
parser.add_argument("-mo", "--momentum", type=float, default=0.9, help="")
parser.add_argument("-w", "--weight_decay", type=float, default=1e-8, help="")
parser.add_argument("-li", "--lr_decay_iters", type=list, default=[1000], help="")
parser.add_argument("-lf", "--lr_decay_factor", type=float, default=0.5, help="")
parser.add_argument("-e", "--epochs", type=int, default=1000, help="")
parser.add_argument("-iu", "--isUnlabel", type=bool, default=False, help="")
parser.add_argument("-ie", "--isEval", type=bool, default=False, help="")
parser.add_argument("-uf", "--useFakePAN", type=bool, default=False, help="")
parser.add_argument("-sensor", "--sensor", type=str, default="", help="")
args = parser.parse_args()
input_data_path = args.testdataDir
input_ms_path = glob.glob(os.path.join(os.path.join(input_data_path, 'MS'), '*.tif'))
input_pan_path = glob.glob(os.path.join(os.path.join(input_data_path, 'PAN'), '*.tif'))
print(input_ms_path, input_pan_path)
input_ms_down_path = glob.glob(os.path.join(os.path.join(input_data_path, 'LR'), '*.tif'))
input_pan_down_path = glob.glob(os.path.join(os.path.join(input_data_path, 'HR'), '*.tif'))

# 创建存储路径
if args.save_result:
    # dl方法

    ours_path = os.path.join(args.saveDir, args.model+args.model_sub+args.model_loss)
    

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
   
    ours_path = os.path.join(args.saveDir, args.model)
 
   
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
if args.model == 'ours':
    model = OursModel()

elif args.model == 'PNN':
    model = PNNModel()

elif args.model == 'PanNet':
    model = PanNetModel()
    
elif args.model == 'TFNet':
    model = TFNetModel()
    
elif args.model == 'MSDCNN':
    model = MSDCNNModel()
   
elif args.model == 'SRPPNN':
    model = SRPPNNModel()


model.initialize(args)
input1 = torch.randn(1, args.mul_channel, 64, 64).cuda()
input2 = torch.randn(1, args.pan_channel, 256, 256).cuda()
flop, para = profile(model.netG, inputs=(input1, input2, ))
print("%.2fM" % (flop/1e6), "%.2fM" % (para/1e6))

print('Total params: %.2fM' % (sum(p.numel() for p in model.netG.parameters())/1000000.0))
model.load_networks(999)

full_times = []
reduce_times = []

ergass = []
sams = []
sccs = []
qs = []
qaves = []
dss = []
d_ls = []
qnrs = []

for i, ms_path in tqdm(enumerate(input_ms_path)):
    # print(i)
    step = i
    
    '''loading data'''
    _, _, used_ms = gdal_read(ms_path)
    geo, proj, used_pan = gdal_read(input_pan_path[i])
    _, _, used_ms_down = gdal_read(input_ms_down_path[i])
    geo_down, proj_down, used_pan_down = gdal_read(input_pan_down_path[i])

    '''normalization'''
    used_ms = used_ms / args.data_range
    used_pan = used_pan / args.data_range
    used_ms_down = used_ms_down / args.data_range
    used_pan_down = used_pan_down / args.data_range

    name = os.path.basename(ms_path)    
    
    if args.isUnlabel:
        fused_image = model.predict(used_ms[:, :, :], used_pan[:, :, :])
        if args.save_result:
            gdal_write(os.path.join(unfull_path,name), fused_image, geo)
        fused_down_image = model.predict(used_ms_down[:, :, :], used_pan_down[:, :, :])
        if args.save_result:
            gdal_write(os.path.join(unreduce_path,name), fused_down_image, geo_down)
    else:
        s_time = time.time()
        fused_image = model.predict(used_ms[:, :, :], used_pan[:, :, :])
        e_time = time.time()
        full_times.append(e_time-s_time)
        # ds = D_s(fused_image, cv2.resize(used_ms, dsize=(0,0),fx=args.scale, fy=args.scale), used_pan, args)
        # d_l = D_lambda(fused_image,  cv2.resize(used_ms, dsize=(0,0),fx=args.scale, fy=args.scale))
        # qnr = QNR(ds, d_l)
        # dss.append(ds)
        # d_ls.append(d_l)
        # qnrs.append(qnr)
        if args.save_result:
            gdal_write(os.path.join(full_path,name), fused_image, geo, proj)

        s_time = time.time()
        fused_down_image = model.predict(used_ms_down[:, :, :], used_pan_down[:, :, :])
        e_time = time.time()
        reduce_times.append(e_time-s_time)
        # ergas = ERGAS(fused_down_image, used_ms)
        # sam = SAM(fused_down_image, used_ms)
        # scc = SCC(fused_down_image, used_ms)
        # q = Q(fused_down_image, used_ms)
        # qave = QAVE(fused_down_image, used_ms)
        # ergass.append(ergas)
        # sams.append(sam)
        # sccs.append(scc)
        # qs.append(q)
        # qaves.append(qave)

        if args.save_result:
           gdal_write(os.path.join(reduce_path,name), fused_down_image, geo_down, proj_down)

full_times = np.array(full_times)
reduce_times = np.array(reduce_times)
ergass = np.array(ergass)
sams = np.array(sams)
sccs = np.array(sccs)
qs = np.array(qs)
qaves = np.array(qaves)
dss = np.array(dss)
d_ls = np.array(d_ls)
qnrs = np.array(qnrs)
print(args.sensor)
print(args.model+args.model_sub+args.model_loss)
print('full_time:{}±{}'.format(np.mean(full_times), np.var(full_times)))
print('reduce_time:{}±{}'.format(np.mean(reduce_times), np.var(reduce_times)))
print('ergas:{}±{}'.format(np.mean(ergass), np.var(ergass)))   
print('sam:{}±{}'.format(np.mean(sams), np.var(sams)))  
print('scc:{}±{}'.format(np.mean(sccs), np.var(sccs)))    
print('q:{}±{}'.format(np.mean(qs), np.var(qs)))
print('qave:{}±{}'.format(np.mean(qaves), np.var(qaves)))
print('ds:{}±{}'.format(np.mean(dss), np.var(dss)))
print('d_l:{}±{}'.format(np.mean(d_ls), np.var(d_ls)))
print('qnr:{}±{}'.format(np.mean(qnrs), np.var(qnrs)))
    





