
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import config as cfg
from scipy import linalg
import kornia
import cupy as cp
import cupyx as cpx
from cupy.core.dlpack import toDlpack
from cupy.core.dlpack import fromDlpack
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
import time
from scipy import ndimage
import cv2
from torchvision.models.vgg import vgg16
from torchvision.models.vgg import vgg19
from pytorch_msssim import SSIM, MS_SSIM
from PIL import Image
# from cvtorchvision import cvtransforms
# from cvtorchvision.cvtransforms import cvfunctional
import numpy as np
# import model.A2B_RCAN as A2B_RCAN

import os

from math import exp
import torch.nn.functional as F
import torch.nn as nn
import torch


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net


def get_scheduler(optimizer):
    if cfg.lr_policy == 'step':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=cfg.lr_decay_iters, gamma=cfg.lr_decay_factor)#StepLR(optimizer, step_size=cfg.lr_decay_iters, gamma=cfg.lr_decay_factor)
    elif cfg.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    return scheduler


class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if cfg.gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif cfg.gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif cfg.gan_mode == 'wgangp':
            self.loss = None

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        if cfg.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = self.loss(input, target_tensor)
        elif cfg.gan_mode == 'wgangp':
            if target_is_real:
                loss = -input.mean()
            else:
                loss = input.mean()
        return loss


class bicubic(nn.Module):
    
    def __init__(self):
        super(bicubic,self).__init__()
    def cubic(self,x):
        absx = torch.abs(x)
        absx2 = torch.abs(x)*torch.abs(x)
        absx3 = torch.abs(x)*torch.abs(x)*torch.abs(x)

        condition1 = (absx<=1).to(torch.float32)
        condition2 = ((1<absx)&(absx<=2)).to(torch.float32)
        
        f = (1.5*absx3 - 2.5*absx2 +1)*condition1+(-0.5*absx3 + 2.5*absx2 -4*absx +2)*condition2
        return f
    def contribute(self,in_size,out_size,scale):
        kernel_width = 4
        if scale<1:
            kernel_width = 4/scale
        x0 = torch.arange(start = 1,end = out_size[0]+1).to(torch.float32)
        x1 = torch.arange(start = 1,end = out_size[1]+1).to(torch.float32)
        
        u0 = x0/scale + 0.5*(1-1/scale)
        u1 = x1/scale + 0.5*(1-1/scale)

        left0 = torch.floor(u0-kernel_width/2)
        left1 = torch.floor(u1-kernel_width/2)

        P = np.ceil(kernel_width)+2
        
        indice0 = left0.unsqueeze(1) + torch.arange(start = 0,end = P).to(torch.float32).unsqueeze(0)
        indice1 = left1.unsqueeze(1) + torch.arange(start = 0,end = P).to(torch.float32).unsqueeze(0)
        
        mid0 = u0.unsqueeze(1) - indice0.unsqueeze(0)
        mid1 = u1.unsqueeze(1) - indice1.unsqueeze(0)

        if scale < 1:
            weight0 = scale* self.cubic(mid0*scale)
            weight1 = scale* self.cubic(mid1*scale)
        else:
            weight0 = self.cubic(mid0)
            weight1 = self.cubic(mid1)

        weight0 = weight0/(torch.sum(weight0,2).unsqueeze(2))
        weight1 = weight1/(torch.sum(weight1,2).unsqueeze(2))

        indice0 = torch.min(torch.max(torch.FloatTensor([1]),indice0),torch.FloatTensor([in_size[0]])).unsqueeze(0)
        indice1 = torch.min(torch.max(torch.FloatTensor([1]),indice1),torch.FloatTensor([in_size[1]])).unsqueeze(0)

        kill0 = torch.eq(weight0,0)[0][0]
        kill1 = torch.eq(weight1,0)[0][0]
        
        weight0 = weight0[:,:,kill0==0]
        weight1 = weight1[:,:,kill1==0]

        indice0 = indice0[:,:,kill0==0]
        indice1 = indice1[:,:,kill1==0]


        return weight0,weight1,indice0,indice1

    def forward(self,input, scale = 1/4):
        # input = kornia.filters.GaussianBlur2d((9,9),(2.1,2.1))(input)
        [b,c,h,w] = input.shape
        #output_size = [b,c,int(h*scale),int(w*scale)]

        weight0,weight1,indice0,indice1 = self.contribute([h,w],[int(h*scale),int(w*scale)],scale)





        weight0 = np.asarray(weight0[0],dtype = np.float32)
        weight0 = torch.from_numpy(weight0).cuda()

        indice0 = np.asarray(indice0[0],dtype = np.float32)
        indice0 = torch.from_numpy(indice0).cuda().long()
        out = input[:,:,(indice0-1),:]*(weight0.unsqueeze(0).unsqueeze(1).unsqueeze(4))
        out = (torch.sum(out,dim = 3))
        A = out.permute(0,1,3,2)

        weight1 = np.asarray(weight1[0],dtype = np.float32)
        weight1 = torch.from_numpy(weight1).cuda()

        indice1 = np.asarray(indice1[0],dtype = np.float32)
        indice1 = torch.from_numpy(indice1).cuda().long()
        out = A[:,:,(indice1-1),:]*(weight1.unsqueeze(0).unsqueeze(1).unsqueeze(4))
        #out = torch.round(255*torch.sum(out,dim = 3).permute(0,1,3,2))/255
        out = torch.sum(out, dim = 3).permute(0,1,3,2)
        #out = kornia.filters.GaussianBlur2d((9,9),(2.1,2.1))(out)
        return out




class S3Loss(nn.Module):
    '''
    Choi, Jae-Seok; Kim, Yongwoo; Kim, Munchurl (2020): S3: A Spectral-Spatial Structure Loss for Pan-Sharpening Networks. 
    In IEEE Geosci. Remote Sensing Lett. 17 (5), pp. 829–833. DOI: 10.1109/LGRS.2019.2934493.
    '''
    def __init__(self, scale, device):
        super(S3Loss, self).__init__()
        self.device = device
        self.scale = scale
        self.L1loss = nn.L1Loss().to(device)
        
        

    def forward(self, lr_ms, hr_pan, hr_ms, fuse_ms):
        down_hr_pan = torch.nn.functional.upsample(hr_pan, scale_factor=1/self.scale, mode='area')

        M_ = torch.zeros(hr_pan.shape).cuda()
        G_ = torch.zeros(hr_pan.shape).cuda()
        for j in range(lr_ms.shape[0]):
            down_hr_pan_array = down_hr_pan[j].view(-1)
            down_hr_pan_array = fromDlpack(to_dlpack(down_hr_pan_array))
            for i in range(lr_ms.shape[1]):
                lr_ms_band = lr_ms[j,i,:,:].squeeze(0)
                lr_ms_band_array = lr_ms_band.view(-1)
                lr_ms_band_array = fromDlpack(to_dlpack(lr_ms_band_array))
                if i == 0:
                    A = cp.vstack([lr_ms_band_array**0, lr_ms_band_array**1])
                else:
                    A = cp.vstack([A,  lr_ms_band_array**1])
            
            sol, r, rank, s = cp.linalg.lstsq(A.T,down_hr_pan_array)
          
            sol = from_dlpack(toDlpack(sol))
            
            for i in range(hr_ms.shape[1]):
                M_[j] += hr_ms[j,i,:,:] * sol[i+1]
                G_[j] += fuse_ms[j,i,:,:] * sol[i+1]
            M_[j] += sol[0]
            G_[j] += sol[0] 
        
        mean_filter = kornia.filters.BoxBlur((31,31))
        e = torch.Tensor([1e-10]).cuda()
        r = 4
        a = 1 
        mean_M_ = mean_filter(M_)
        mean_P =  mean_filter(hr_pan) 
        mean_M_xP = mean_filter(M_*hr_pan)
        cov_M_xP = mean_M_xP - mean_M_*mean_P
        mean_M_xM_ = mean_filter(M_*M_)
        std_M_ = torch.sqrt(torch.abs(mean_M_xM_ - mean_M_*mean_M_) + e)
        mean_PxP = mean_filter(hr_pan*hr_pan)
        std_P = torch.sqrt(torch.abs(mean_PxP - mean_P*mean_P) + e)
        corr_M_xP = cov_M_xP / (std_M_*std_P)
        S = corr_M_xP**r
        loss_c = self.L1loss(fuse_ms*S, hr_ms*S)

        grad_P = (hr_pan - mean_P) / std_P
        mean_G_ = mean_filter(G_)
        mean_G_xG_ = mean_filter(G_*G_)
        std_G_ = torch.sqrt(torch.abs(mean_G_xG_ - mean_G_*mean_G_) + e)
        grad_G_ = (G_ - mean_G_) / std_G_
        loss_a = self.L1loss(grad_G_*(2-S), grad_P*(2-S))

        loss = loss_c + a*loss_a
        return loss
        




class OursLoss2(nn.Module):
    
    def __init__(self, scale, device):
        super(OursLoss2, self).__init__()
        self.device = device
        self.scale = scale
        self.mseloss = nn.MSELoss().to(device)
       

    def forward(self, lr_ms, hr_pan, fuse_ms, hr_ms):
        loss1 = self.mseloss(fuse_ms, hr_ms)
 
        return loss1
        
        

       

class DIPLoss(nn.Module):
    def __init__(self, device):
        super(DIPLoss, self).__init__()

        self.ssimmulloss = SSIM(data_range=1, channel=cfg.mul_channel).to(device)
        
    def forward(self, lr_ms, hr_pan, fuse_ms, hr_ms):
      
        loss_whole =1 - self.ssimmulloss(fuse_ms, hr_ms)
        
        return loss_whole



class L1Loss(nn.Module):
    def __init__(self, device):
        super(L1Loss, self).__init__()

        self.l1_loss = nn.L1Loss().to(device)

    def forward(self, lr_ms, hr_pan, fuse_ms, hr_ms):
        
        loss_whole =self.l1_loss(fuse_ms, hr_ms)

        return loss_whole
