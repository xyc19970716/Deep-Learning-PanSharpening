
import os
import itertools
from collections import OrderedDict
import torch.nn as nn
import torch
from .base_model import BaseModel
from . import networks
import config as cfg
import math
import numpy as np
import model.optim as optim
from torch.autograd import Function, Variable
import kornia
import cupy as cp
from cupy.core.dlpack import toDlpack
from cupy.core.dlpack import fromDlpack
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack


class Residual_Block(nn.Module):
    def __init__(self, channels):
        super(Residual_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
       
    def forward(self, x):
        fea = self.relu(self.bn1(self.conv1(x)))
        fea = self.relu(self.bn2(self.conv2(fea)))
        result = fea + x
        return result



def default_conv(in_channels, out_channels, kernel_size, padding=0, stride =1,bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)





class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class SAU(nn.Module): 
    def __init__(self, nFeat, conv=nn.Conv2d):
        super(SAU, self).__init__()
        self.down_channel = conv(nFeat * 3, nFeat, kernel_size=1, padding=0, bias=True)

    def forward(self, high, low):# pan mul
        
        concat = torch.cat([low, high], -3)
        mul = torch.add(low, high)
        concat = torch.cat([concat, mul], -3)
       
        output = self.down_channel(concat)
        return output



class A2B_RCAN1(nn.Module): # 2020 0722
    def __init__(self, block_num=2, scale=4, nFeat=64, conv=nn.Conv2d):
        super(A2B_RCAN1, self).__init__()

        self.upsample1 = Upsampler(conv=default_conv, scale=2, n_feats=nFeat, bn=False, act='relu') # act 使用激活函数relu bn改为true
        self.upsample2 = Upsampler(conv=default_conv, scale=2, n_feats=nFeat, bn=False, act='relu') # act 使用激活函数relu bn改为true
        self.mul_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=cfg.mul_channel, out_channels=nFeat, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2), 
        ) 
        self.sau_l1 = SAU(nFeat)

        self.sau_l2 = SAU(nFeat)
      
        self.sau_h1 = SAU(nFeat)

        self.sau_h2 = SAU(nFeat)
      
        
        self.out = nn.Sequential(
            nn.Conv2d(nFeat, cfg.mul_channel, kernel_size=1, padding=0),
      
        ) 


        self.pan_hp_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=cfg.pan_channel, out_channels=nFeat, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        ) 
        self.pan_hp_res_1 = nn.Sequential(
            self.make_layer(Residual_Block, block_num, nFeat),
        ) 
        self.pan_hp_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=nFeat, out_channels=nFeat, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.pan_hp_res_2 = nn.Sequential(
            self.make_layer(Residual_Block, block_num, nFeat),
        ) 
        self.pan_hp_conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=nFeat, out_channels=nFeat, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.pan_hp_res_3 = nn.Sequential(
            self.make_layer(Residual_Block, block_num, nFeat),
        ) 
        self.pan_lp_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=cfg.pan_channel, out_channels=nFeat, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.pan_lp_res_1 = nn.Sequential(
            self.make_layer(Residual_Block, block_num, nFeat),
        ) 
        self.pan_lp_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=nFeat, out_channels=nFeat, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.pan_lp_res_2 = nn.Sequential(
            self.make_layer(Residual_Block, block_num, nFeat),
        ) 
        self.pan_lp_conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=nFeat, out_channels=nFeat, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.pan_lp_res_3 = nn.Sequential(
            self.make_layer(Residual_Block, block_num, nFeat),
        ) 
        self.perup1 = self.make_layer(Residual_Block, block_num, nFeat)
        self.perup2 = self.make_layer(Residual_Block, block_num, nFeat)
        
   

      

        self.encode2 = nn.Sequential(
            nn.Conv2d(in_channels=nFeat, out_channels=nFeat, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.encode3 = nn.Sequential(
            nn.Conv2d(in_channels=nFeat, out_channels=nFeat*2, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.encode4 = nn.Sequential(
            nn.Conv2d(in_channels=nFeat*2, out_channels=nFeat*4, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )

       

        self.decode1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=nFeat*(4+2), out_channels=nFeat*3, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(0.2),
        )

        self.decode2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=nFeat*(3+2+2), out_channels=nFeat*3, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(0.2),
        )

        self.decode3 = nn.Sequential(
            nn.Conv2d(in_channels=nFeat*(3+1+2), out_channels=nFeat, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.bicubic = networks.bicubic()

    

       

    def make_layer(self, block, num_of_layer, channels):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(channels))
        return nn.Sequential(*layers)

    def forward(self, x, y):
        x_up = self.bicubic(x, scale=cfg.scale)#torch.nn.functional.upsample(x, scale_factor=cfg.scale, mode='bicubic')
      
        x = self.mul_conv_1(x)

        y_lp = kornia.filters.GaussianBlur2d((11,11),(1,1))(y)
        y_hp = y - y_lp

       
        y_1_hp = self.pan_hp_conv_1(y_hp)
        y_1_hp = self.pan_hp_res_1(y_1_hp) + y_1_hp
   
        y_2_hp = self.pan_hp_conv_2(y_1_hp)
        y_2_hp = self.pan_hp_res_2(y_2_hp) + y_2_hp
    
        y_3_hp = self.pan_hp_conv_3(y_2_hp)
        y_3_hp = self.pan_hp_res_3(y_3_hp) + y_3_hp


        y_1_lp = self.pan_lp_conv_1(y_lp)
        y_1_lp = self.pan_lp_res_1(y_1_lp) + y_1_lp

        y_2_lp = self.pan_lp_conv_2(y_1_lp)
        y_2_lp = self.pan_lp_res_2(y_2_lp) + y_2_lp
 
        y_3_lp = self.pan_lp_conv_3(y_2_lp)
        y_3_lp = self.pan_lp_res_3(y_3_lp) + y_3_lp


        
        x = self.sau_l1(x, y_3_lp)
        
        x = self.perup1(x) + x
        
        x = self.upsample1(x) + y_2_hp

   
        x = self.sau_l2(x, y_2_lp)
        x = self.perup2(x) + x
      
        x = self.upsample2(x) + y_1_hp
 
  
        e2 = self.encode2(x)
        e3 = self.encode3(e2)
        e4 = self.encode4(e3) 
        
        d1 = self.decode1(torch.cat([e4, y_3_lp, y_3_hp], -3))
        d2 = self.decode2(torch.cat([d1, e3, y_2_lp, y_2_hp], -3))
        d3 = self.decode3(torch.cat([d2, e2, y_1_lp, y_1_hp], -3))
     
        x = self.out(d3) + x_up 
        
        return x



class A2B_RCAN1_1(nn.Module):
    def __init__(self, block_num=1, scale=4, nFeat=64, conv=nn.Conv2d):
        super(A2B_RCAN1_1, self).__init__()

        self.upsample1 = Upsampler(conv=default_conv, scale=2, n_feats=nFeat, bn=False, act='relu') # act 使用激活函数relu bn改为true
        self.upsample2 = Upsampler(conv=default_conv, scale=2, n_feats=nFeat, bn=False, act='relu') # act 使用激活函数relu bn改为true
        self.mul_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=cfg.mul_channel, out_channels=nFeat, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2), 
        ) 
        self.sau_l1 = SAU(nFeat)

        self.sau_l2 = SAU(nFeat)
      
        self.sau_h1 = SAU(nFeat)

        self.sau_h2 = SAU(nFeat)
      
        
        self.out = nn.Sequential(
            nn.Conv2d(nFeat, cfg.mul_channel, kernel_size=1, padding=0),
      
        ) 
        self.activation = nn.Tanh()

        self.pan_hp_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=cfg.pan_channel, out_channels=nFeat, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        ) 
        self.pan_hp_res_1 = nn.Sequential(
            self.make_layer(Residual_Block, block_num, nFeat),
        ) 
        self.pan_hp_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=nFeat, out_channels=nFeat, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.pan_hp_res_2 = nn.Sequential(
            self.make_layer(Residual_Block, block_num, nFeat),
        ) 
        self.pan_hp_conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=nFeat, out_channels=nFeat, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.pan_hp_res_3 = nn.Sequential(
            self.make_layer(Residual_Block, block_num, nFeat),
        ) 
        self.pan_lp_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=cfg.pan_channel, out_channels=nFeat, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.pan_lp_res_1 = nn.Sequential(
            self.make_layer(Residual_Block, block_num, nFeat),
        ) 
        self.pan_lp_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=nFeat, out_channels=nFeat, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.pan_lp_res_2 = nn.Sequential(
            self.make_layer(Residual_Block, block_num, nFeat),
        ) 
        self.pan_lp_conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=nFeat, out_channels=nFeat, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.pan_lp_res_3 = nn.Sequential(
            self.make_layer(Residual_Block, block_num, nFeat),
        ) 
        self.perup1 = self.make_layer(Residual_Block, block_num, nFeat)
        self.perup2 = self.make_layer(Residual_Block, block_num, nFeat)
        
      
      

        self.encode2 = nn.Sequential(
            nn.Conv2d(in_channels=nFeat, out_channels=nFeat, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.encode3 = nn.Sequential(
            nn.Conv2d(in_channels=nFeat, out_channels=nFeat*2, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.encode4 = nn.Sequential(
            nn.Conv2d(in_channels=nFeat*2, out_channels=nFeat*4, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )

       

        self.decode1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=nFeat*(4+2), out_channels=nFeat*3, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(0.2),
        )

        self.decode2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=nFeat*(3+2+2), out_channels=nFeat*3, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(0.2),
        )

        self.decode3 = nn.Sequential(
            nn.Conv2d(in_channels=nFeat*(3+1+2), out_channels=nFeat, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.bicubic = networks.bicubic()

       

    def make_layer(self, block, num_of_layer, channels):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(channels))
        return nn.Sequential(*layers)

    def forward(self, x, y):
        x_up = self.bicubic(x, scale=cfg.scale)
      
        x = self.mul_conv_1(x)

      
        y_lp = kornia.filters.GaussianBlur2d((11,11),(1,1))(y)
        y_hp = y - y_lp

      
        y_1_hp = self.pan_hp_conv_1(y_hp)
        y_1_hp = self.pan_hp_res_1(y_1_hp)

        y_2_hp = self.pan_hp_conv_2(y_1_hp)
        y_2_hp = self.pan_hp_res_2(y_2_hp)

        y_3_hp = self.pan_hp_conv_3(y_2_hp)
        y_3_hp = self.pan_hp_res_3(y_3_hp)


        y_1_lp = self.pan_lp_conv_1(y_lp)
        y_1_lp = self.pan_lp_res_1(y_1_lp)

        y_2_lp = self.pan_lp_conv_2(y_1_lp)
        y_2_lp = self.pan_lp_res_2(y_2_lp)

        y_3_lp = self.pan_lp_conv_3(y_2_lp)
        y_3_lp = self.pan_lp_res_3(y_3_lp) 
        
        
        x = self.sau_l1(x, y_3_lp)
        
        x = self.perup1(x)# + x
        
        x = self.upsample1(x) + y_2_hp

        x = self.sau_l2(x, y_2_lp)
        x = self.perup2(x)# + x

        x = self.upsample2(x) + y_1_hp
 
  
       
        e2 = self.encode2(x)
        e3 = self.encode3(e2)
        e4 = self.encode4(e3) 
        
        d1 = self.decode1(torch.cat([e4, y_3_lp, y_3_hp], -3))
        d2 = self.decode2(torch.cat([d1, e3, y_2_lp, y_2_hp], -3))
        d3 = self.decode3(torch.cat([d2, e2, y_1_lp, y_1_hp], -3))
     
        x = self.out(d3) + x_up
        
        return x

class A2B_RCAN1_0(nn.Module): 
    def __init__(self, block_num=1, scale=4, nFeat=64, conv=nn.Conv2d):
        super(A2B_RCAN1_0, self).__init__()

        self.upsample1 = Upsampler(conv=default_conv, scale=2, n_feats=nFeat, bn=False, act='relu') # act 使用激活函数relu bn改为true
        self.upsample2 = Upsampler(conv=default_conv, scale=2, n_feats=nFeat, bn=False, act='relu') # act 使用激活函数relu bn改为true
        self.mul_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=cfg.mul_channel, out_channels=nFeat, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2), 
        ) 
        self.sau_l1 = SAU(nFeat)

        self.sau_l2 = SAU(nFeat)
      
        self.sau_h1 = SAU(nFeat)

        self.sau_h2 = SAU(nFeat)
      
        
        self.out = nn.Sequential(
            nn.Conv2d(nFeat, cfg.mul_channel, kernel_size=1, padding=0),
        ) 
        self.activation = nn.Tanh()

        self.pan_hp_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=cfg.pan_channel, out_channels=nFeat, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        ) 
     
        self.pan_hp_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=nFeat, out_channels=nFeat, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        
        self.pan_hp_conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=nFeat, out_channels=nFeat, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        
        self.pan_lp_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=cfg.pan_channel, out_channels=nFeat, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        )
       
        self.pan_lp_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=nFeat, out_channels=nFeat, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        
        self.pan_lp_conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=nFeat, out_channels=nFeat, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        
      
      

        self.encode2 = nn.Sequential(
            nn.Conv2d(in_channels=nFeat, out_channels=nFeat, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.encode3 = nn.Sequential(
            nn.Conv2d(in_channels=nFeat, out_channels=nFeat*2, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.encode4 = nn.Sequential(
            nn.Conv2d(in_channels=nFeat*2, out_channels=nFeat*4, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )

       

        self.decode1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=nFeat*(4+2), out_channels=nFeat*3, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(0.2),
        )

        self.decode2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=nFeat*(3+2+2), out_channels=nFeat*3, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(0.2),
        )

        self.decode3 = nn.Sequential(
            nn.Conv2d(in_channels=nFeat*(3+1+2), out_channels=nFeat, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.bicubic = networks.bicubic()


       

    def make_layer(self, block, num_of_layer, channels):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(channels))
        return nn.Sequential(*layers)

    def forward(self, x, y):
        x_up = self.bicubic(x, scale=cfg.scale)
      
        x = self.mul_conv_1(x)

      
        y_lp = kornia.filters.GaussianBlur2d((11,11),(1,1))(y)
        y_hp = y - y_lp

      
        y_1_hp = self.pan_hp_conv_1(y_hp)


        y_2_hp = self.pan_hp_conv_2(y_1_hp)
  

        y_3_hp = self.pan_hp_conv_3(y_2_hp)



        y_1_lp = self.pan_lp_conv_1(y_lp)
  

        y_2_lp = self.pan_lp_conv_2(y_1_lp)

        y_3_lp = self.pan_lp_conv_3(y_2_lp)
 
        
        
        x = self.sau_l1(x, y_3_lp)
        
        
        
        x = self.upsample1(x) + y_2_hp

        x = self.sau_l2(x, y_2_lp)
      

        x = self.upsample2(x) + y_1_hp
 

        e2 = self.encode2(x)
        e3 = self.encode3(e2)
        e4 = self.encode4(e3) 
        
        d1 = self.decode1(torch.cat([e4, y_3_lp, y_3_hp], -3))
        d2 = self.decode2(torch.cat([d1, e3, y_2_lp, y_2_hp], -3))
        d3 = self.decode3(torch.cat([d2, e2, y_1_lp, y_1_hp], -3))
     
        x = self.out(d3) + x_up
        
        return x


class A2B_RCAN1_NOFFU(nn.Module):
    def __init__(self, block_num=2, scale=4, nFeat=64, conv=nn.Conv2d):
        super(A2B_RCAN1_NOFFU, self).__init__()

        self.upsample1 = Upsampler(conv=default_conv, scale=2, n_feats=nFeat, bn=False, act='relu') # act 使用激活函数relu bn改为true
        self.upsample2 = Upsampler(conv=default_conv, scale=2, n_feats=nFeat, bn=False, act='relu') # act 使用激活函数relu bn改为true
        self.mul_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=cfg.mul_channel, out_channels=nFeat, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2), 
        ) 
        
      
        
        self.out = nn.Sequential(
            nn.Conv2d(nFeat, cfg.mul_channel, kernel_size=1, padding=0),

        ) 
        self.activation = nn.Tanh()

      
        self.pan_hp_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=cfg.pan_channel, out_channels=nFeat, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        ) 
        self.pan_hp_res_1 = nn.Sequential(
            self.make_layer(Residual_Block, block_num, nFeat),
        ) 
        self.pan_hp_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=nFeat, out_channels=nFeat, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.pan_hp_res_2 = nn.Sequential(
            self.make_layer(Residual_Block, block_num, nFeat),
        ) 
        self.pan_hp_conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=nFeat, out_channels=nFeat, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.pan_hp_res_3 = nn.Sequential(
            self.make_layer(Residual_Block, block_num, nFeat),
        ) 
        self.pan_lp_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=cfg.pan_channel, out_channels=nFeat, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.pan_lp_res_1 = nn.Sequential(
            self.make_layer(Residual_Block, block_num, nFeat),
        ) 
        self.pan_lp_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=nFeat, out_channels=nFeat, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.pan_lp_res_2 = nn.Sequential(
            self.make_layer(Residual_Block, block_num, nFeat),
        ) 
        self.pan_lp_conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=nFeat, out_channels=nFeat, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.pan_lp_res_3 = nn.Sequential(
            self.make_layer(Residual_Block, block_num, nFeat),
        ) 
        self.perup1 = self.make_layer(Residual_Block, block_num, nFeat)
        self.perup2 = self.make_layer(Residual_Block, block_num, nFeat)
        
     
      

        self.encode2 = nn.Sequential(
            nn.Conv2d(in_channels=nFeat, out_channels=nFeat, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.encode3 = nn.Sequential(
            nn.Conv2d(in_channels=nFeat, out_channels=nFeat*2, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.encode4 = nn.Sequential(
            nn.Conv2d(in_channels=nFeat*2, out_channels=nFeat*4, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )

       

        self.decode1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=nFeat*(4+2), out_channels=nFeat*3, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(0.2),
        )

        self.decode2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=nFeat*(3+2+2), out_channels=nFeat*3, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(0.2),
        )

        self.decode3 = nn.Sequential(
            nn.Conv2d(in_channels=nFeat*(3+1+2), out_channels=nFeat, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.bicubic = networks.bicubic()


       

    def make_layer(self, block, num_of_layer, channels):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(channels))
        return nn.Sequential(*layers)

    def forward(self, x, y):
        x_up = self.bicubic(x, scale=cfg.scale)
       
        x = self.mul_conv_1(x)

        y_lp = kornia.filters.GaussianBlur2d((11,11),(1,1))(y)
        y_hp = y - y_lp

        y_1_hp = self.pan_hp_conv_1(y_hp)
        y_1_hp = self.pan_hp_res_1(y_1_hp) + y_1_hp

        y_2_hp = self.pan_hp_conv_2(y_1_hp)
        y_2_hp = self.pan_hp_res_2(y_2_hp) + y_2_hp

        y_3_hp = self.pan_hp_conv_3(y_2_hp)
        y_3_hp = self.pan_hp_res_3(y_3_hp) + y_3_hp


        y_1_lp = self.pan_lp_conv_1(y_lp)
        y_1_lp = self.pan_lp_res_1(y_1_lp) + y_1_lp

        y_2_lp = self.pan_lp_conv_2(y_1_lp)
        y_2_lp = self.pan_lp_res_2(y_2_lp) + y_2_lp

        y_3_lp = self.pan_lp_conv_3(y_2_lp)
        y_3_lp = self.pan_lp_res_3(y_3_lp) + y_3_lp

        x = x+y_3_lp
        
        x = self.perup1(x) + x
        
        x = self.upsample1(x) + y_2_hp


        x = x+y_2_lp
        x = self.perup2(x) + x



        x = self.upsample2(x) + y_1_hp
 
  

        e2 = self.encode2(x)
        e3 = self.encode3(e2)
        e4 = self.encode4(e3) 
        
        d1 = self.decode1(torch.cat([e4, y_3_lp, y_3_hp], -3))
        d2 = self.decode2(torch.cat([d1, e3, y_2_lp, y_2_hp], -3))
        d3 = self.decode3(torch.cat([d2, e2, y_1_lp, y_1_hp], -3))
     
        x = self.out(d3) + x_up 
        
        return x

      
class SR_PLB(nn.Module): 
    def __init__(self, scale=4, nFeat=64, conv=nn.Conv2d):
        super(SR_PLB, self).__init__()

        self.upsample1 = Upsampler(conv=default_conv, scale=2, n_feats=nFeat, bn=False, act='relu') # act 使用激活函数relu bn改为true
        self.upsample2 = Upsampler(conv=default_conv, scale=2, n_feats=nFeat, bn=False, act='relu') # act 使用激活函数relu bn改为true
        self.mul_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=cfg.mul_channel, out_channels=nFeat, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2), 
        ) 
        self.sau_l1 = SAU(nFeat)

        self.sau_l2 = SAU(nFeat)

        
        self.out = nn.Sequential(
            nn.Conv2d(nFeat, cfg.mul_channel, kernel_size=3, padding=1),
      
        ) 
        self.activation = nn.Tanh()
        
        self.pan_lp_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=cfg.pan_channel, out_channels=nFeat, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.pan_lp_res_1 = nn.Sequential(
            self.make_layer(Residual_Block, 2, nFeat),
        ) 
        self.pan_lp_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=nFeat, out_channels=nFeat, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.pan_lp_res_2 = nn.Sequential(
            self.make_layer(Residual_Block, 2, nFeat),
        ) 
        self.pan_lp_conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=nFeat, out_channels=nFeat, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.pan_lp_res_3 = nn.Sequential(
            self.make_layer(Residual_Block, 2, nFeat),
        ) 
        self.perup1 = self.make_layer(Residual_Block, 2, nFeat)
        self.perup2 = self.make_layer(Residual_Block, 2, nFeat)
  

        self.bicubic = networks.bicubic()


    def make_layer(self, block, num_of_layer, channels):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(channels))
        return nn.Sequential(*layers)

    def forward(self, x, y):
        x_up = self.bicubic(x, scale=cfg.scale)
       
        x = self.mul_conv_1(x)
      
        y_lp = kornia.filters.GaussianBlur2d((11,11),(1,1))(y)
        y_hp = y - y_lp

        y_1_lp = self.pan_lp_conv_1(y_lp)
        y_1_lp = self.pan_lp_res_1(y_1_lp) + y_1_lp

        y_2_lp = self.pan_lp_conv_2(y_1_lp)
        y_2_lp = self.pan_lp_res_2(y_2_lp) + y_2_lp

        y_3_lp = self.pan_lp_conv_3(y_2_lp)
        y_3_lp = self.pan_lp_res_3(y_3_lp) + y_3_lp

        x = self.sau_l1(x, y_3_lp)
        
        x = self.perup1(x) + x

        x = self.upsample1(x) 
       
        x = self.sau_l2(x, y_2_lp)

        x = self.perup2(x) + x

        x = self.upsample2(x)
      
       
        
        x = self.out(x) + x_up
        
        
        return x

class SR_PHB(nn.Module): 
    def __init__(self, scale=4, nFeat=64, conv=nn.Conv2d):
        super(SR_PHB, self).__init__()

        self.upsample1 = Upsampler(conv=default_conv, scale=2, n_feats=nFeat, bn=False, act='relu') # act 使用激活函数relu bn改为true
        self.upsample2 = Upsampler(conv=default_conv, scale=2, n_feats=nFeat, bn=False, act='relu') # act 使用激活函数relu bn改为true
        self.mul_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=cfg.mul_channel, out_channels=nFeat, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2), 
        ) 
      

        
        self.out = nn.Sequential(
            nn.Conv2d(nFeat, cfg.mul_channel, kernel_size=3, padding=1),

        ) 
        self.activation = nn.Tanh()
        
        self.pan_lp_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=cfg.pan_channel, out_channels=nFeat, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.pan_lp_res_1 = nn.Sequential(
            self.make_layer(Residual_Block, 2, nFeat),
        ) 
        self.pan_lp_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=nFeat, out_channels=nFeat, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.pan_lp_res_2 = nn.Sequential(
            self.make_layer(Residual_Block, 2, nFeat),
        ) 
        self.pan_lp_conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=nFeat, out_channels=nFeat, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.pan_lp_res_3 = nn.Sequential(
            self.make_layer(Residual_Block, 2, nFeat),
        ) 
        self.perup1 = self.make_layer(Residual_Block, 2, nFeat)
        self.perup2 = self.make_layer(Residual_Block, 2, nFeat)
  

        self.bicubic = networks.bicubic()


    def make_layer(self, block, num_of_layer, channels):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(channels))
        return nn.Sequential(*layers)

    def forward(self, x, y):
        x_up = self.bicubic(x, scale=cfg.scale)
        

        x = self.mul_conv_1(x)
      
        y_lp = kornia.filters.GaussianBlur2d((11,11),(1,1))(y)
        y_lp = y - y_lp

        y_1_lp = self.pan_lp_conv_1(y_lp)
        y_1_lp = self.pan_lp_res_1(y_1_lp) + y_1_lp

        y_2_lp = self.pan_lp_conv_2(y_1_lp)
        y_2_lp = self.pan_lp_res_2(y_2_lp) + y_2_lp

        y_3_lp = self.pan_lp_conv_3(y_2_lp)
        y_3_lp = self.pan_lp_res_3(y_3_lp) + y_3_lp

 
        
        x = self.perup1(x) + x

        x = self.upsample1(x) + y_2_lp
       
 

        x = self.perup2(x) + x

        x = self.upsample2(x) +y_1_lp
      
       
        
        x = self.out(x) + x_up
        
        
        return x


class SR(nn.Module):
    def __init__(self, scale=4, nFeat=64, conv=nn.Conv2d):
        super(SR, self).__init__()

        self.upsample1 = Upsampler(conv=default_conv, scale=2, n_feats=nFeat, bn=False, act='relu') # act 使用激活函数relu bn改为true
        self.upsample2 = Upsampler(conv=default_conv, scale=2, n_feats=nFeat, bn=False, act='relu') # act 使用激活函数relu bn改为true
        self.mul_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=cfg.mul_channel, out_channels=nFeat, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2), 
        ) 
       
        self.out = nn.Sequential(
            nn.Conv2d(nFeat, cfg.mul_channel, kernel_size=3, padding=1),
  
        ) 
        self.activation = nn.Tanh()
        
        self.perup1 = self.make_layer(Residual_Block, 2, nFeat)
        self.perup2 = self.make_layer(Residual_Block, 2, nFeat)
  

        self.bicubic = networks.bicubic()

    def make_layer(self, block, num_of_layer, channels):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(channels))
        return nn.Sequential(*layers)

    def forward(self, x, y):
        x_up = self.bicubic(x, scale=cfg.scale)

        x = self.mul_conv_1(x)
      
        x = self.perup1(x) + x

        x = self.upsample1(x) 
        
        x = self.perup2(x) + x

        x = self.upsample2(x)
       
        x = self.out(x) + x_up
        
        return x

class SR_PLB_PHB(nn.Module):
    def __init__(self, scale=4, nFeat=64, conv=nn.Conv2d):
        super(SR_PLB_PHB, self).__init__()

        self.upsample1 = Upsampler(conv=default_conv, scale=2, n_feats=nFeat, bn=False, act='relu') # act 使用激活函数relu bn改为true
        self.upsample2 = Upsampler(conv=default_conv, scale=2, n_feats=nFeat, bn=False, act='relu') # act 使用激活函数relu bn改为true
        self.mul_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=cfg.mul_channel, out_channels=nFeat, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2), 
        ) 
        self.sau_l1 = SAU(nFeat)

        self.sau_l2 = SAU(nFeat)
      

        
        self.out = nn.Sequential(
            nn.Conv2d(nFeat, cfg.mul_channel, kernel_size=3, padding=1),
           
        ) 
        self.activation = nn.Tanh()
        self.pan_hp_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=cfg.pan_channel, out_channels=nFeat, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        ) 
        self.pan_hp_res_1 = nn.Sequential(
            self.make_layer(Residual_Block, 2, nFeat),
        ) 
        self.pan_hp_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=nFeat, out_channels=nFeat, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.pan_hp_res_2 = nn.Sequential(
            self.make_layer(Residual_Block, 2, nFeat),
        ) 
        self.pan_hp_conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=nFeat, out_channels=nFeat, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.pan_hp_res_3 = nn.Sequential(
            self.make_layer(Residual_Block, 2, nFeat),
        ) 
        self.pan_lp_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=cfg.pan_channel, out_channels=nFeat, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.pan_lp_res_1 = nn.Sequential(
            self.make_layer(Residual_Block, 2, nFeat),
        ) 
        self.pan_lp_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=nFeat, out_channels=nFeat, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.pan_lp_res_2 = nn.Sequential(
            self.make_layer(Residual_Block, 2, nFeat),
        ) 
        self.pan_lp_conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=nFeat, out_channels=nFeat, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.pan_lp_res_3 = nn.Sequential(
            self.make_layer(Residual_Block, 2, nFeat),
        ) 
        self.perup1 = self.make_layer(Residual_Block, 2, nFeat)
        self.perup2 = self.make_layer(Residual_Block, 2, nFeat)
    

        self.bicubic = networks.bicubic()


    def make_layer(self, block, num_of_layer, channels):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(channels))
        return nn.Sequential(*layers)

    def forward(self, x, y):
        x_up = self.bicubic(x, scale=cfg.scale)
       

        x = self.mul_conv_1(x)
      
        y_lp = kornia.filters.GaussianBlur2d((11,11),(1,1))(y)
        y_hp = y - y_lp


        y_1_hp = self.pan_hp_conv_1(y_hp)
        y_1_hp = self.pan_hp_res_1(y_1_hp) + y_1_hp

        y_2_hp = self.pan_hp_conv_2(y_1_hp)
        y_2_hp = self.pan_hp_res_2(y_2_hp) + y_2_hp

        y_3_hp = self.pan_hp_conv_3(y_2_hp)
        y_3_hp = self.pan_hp_res_3(y_3_hp) + y_3_hp


        y_1_lp = self.pan_lp_conv_1(y_lp)
        y_1_lp = self.pan_lp_res_1(y_1_lp) + y_1_lp

        y_2_lp = self.pan_lp_conv_2(y_1_lp)
        y_2_lp = self.pan_lp_res_2(y_2_lp) + y_2_lp

        y_3_lp = self.pan_lp_conv_3(y_2_lp)
        y_3_lp = self.pan_lp_res_3(y_3_lp) + y_3_lp
 

        x = self.sau_l1(x, y_3_lp)
        
        x = self.perup1(x) + x
        
        x = self.upsample1(x) + y_2_hp

        x = self.sau_l2(x, y_2_lp)

        x = self.perup2(x) + x
    
        x = self.upsample2(x) + y_1_hp
      
        x = self.out(x) + x_up
        
        return x


class OursModel(BaseModel):
    
    def initialize(self):
        BaseModel.initialize(self)
      
        self.save_dir = os.path.join(cfg.checkpoints_dir, cfg.model+cfg.model_sub+cfg.model_loss) # 定义checkpoints路径
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print("Create file path: ", self.save_dir)

        if cfg.isUnlabel:
            self.save_dir = os.path.join(self.save_dir, 'unsupervised')
        else:
            self.save_dir = os.path.join(self.save_dir, 'supervised')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print("Create file path: ", self.save_dir)

        self.loss_names = ['G']

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G']
        else:  # during test time, only load Gs
            self.model_names = ['G']

        # load/define networks
        if cfg.model_sub == 'SR':
            self.netG = networks.init_net(SR()).cuda()
        elif cfg.model_sub == 'SR_PLB':
            self.netG = networks.init_net(SR_PLB()).cuda()
        elif cfg.model_sub == 'SR_PHB':
            self.netG = networks.init_net(SR_PHB()).cuda()
        elif cfg.model_sub == 'SR_PLB_PHB':
            self.netG = networks.init_net(SR_PLB_PHB()).cuda()
        elif cfg.model_sub == 'NOFFU':
            self.netG = networks.init_net(A2B_RCAN1_NOFFU()).cuda()
        elif cfg.model_sub=='16':
            self.netG = networks.init_net(A2B_RCAN1(nFeat=16)).cuda()
        elif cfg.model_sub=='32':
            self.netG = networks.init_net(A2B_RCAN1(nFeat=32)).cuda()
        elif cfg.model_sub=='48':
            self.netG = networks.init_net(A2B_RCAN1(nFeat=48)).cuda()
        elif cfg.model_sub == 'RES1':
            self.netG = networks.init_net(A2B_RCAN1_1()).cuda()
        elif cfg.model_sub == 'RES0':
            self.netG = networks.init_net(A2B_RCAN1_0()).cuda()
        elif cfg.model_sub == 'RES3':
            self.netG = networks.init_net(A2B_RCAN1(block_num=3)).cuda()
        else:
            self.netG = networks.init_net(A2B_RCAN1()).cuda()
      

        if self.isTrain:
            
            
            # define loss functions
            
            if cfg.isUnlabel:
                self.criterionL1 = networks.OursLoss(scale=cfg.scale, device=self.device)#
            else:
                if cfg.model_sub=='DIP':
                    # self.criterionL1 = networks.S3Loss(scale=cfg.scale, device=self.device)
                    self.criterionL1 = networks.DIPLoss(device=self.device)
                elif cfg.model_loss=='SSIM':
                    # self.criterionL1 = networks.S3Loss(scale=cfg.scale, device=self.device)
                    self.criterionL1 = networks.DIPLoss(device=self.device)
                elif cfg.model_loss=='L1':
                    self.criterionL1 = networks.L1Loss(device=self.device)#torch.nn.L1Loss()
                else:
                    self.criterionL1 = networks.OursLoss2(scale=cfg.scale, device=self.device)#
            # initialize optimizers
    
            if cfg.optim_type=='adam':
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=cfg.lr, betas=(cfg.beta, 0.999), weight_decay=cfg.weight_decay)
            elif cfg.optim_type=='sgd':
                self.optimizer_G = torch.optim.SGD(self.netG.parameters(),
                                                    lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
      
        

    def set_input(self, input_dict):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        if cfg.isUnlabel:
            self.real_A_1 = input_dict['A_1'].to(self.device)  # mul
            self.real_A_2 = input_dict['A_2'].to(self.device)  # pan
        else:
            self.real_A_1 = input_dict['A_1'].to(self.device)  # mul
            self.real_A_2 = input_dict['A_2'].to(self.device)  # pan
            self.real_B = input_dict['B'].to(self.device) # fus
          

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if cfg.isUnlabel:
            self.fake_B, self.fake_pan = self.netG(self.real_A_1, self.real_A_2) 
        else:
            self.fake_B = self.netG(self.real_A_1, self.real_A_2) 

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        #
       
        if cfg.isUnlabel:
            self.loss_G = self.criterionL1(self.real_A_1, self.real_A_2,self.fake_B, self.fake_pan)
        else:
           
            self.loss_G = self.criterionL1(self.real_A_1, self.real_A_2,self.fake_B, self.real_B)
                # self.loss_G = self.criterionL1(self.fake_B, self.real_B) 
   
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights