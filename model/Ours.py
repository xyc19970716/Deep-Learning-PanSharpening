
import os
import itertools
from collections import OrderedDict
import torch.nn as nn
import torch
from .base_model import BaseModel
from . import networks
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
import torch.nn.functional as F
from functools import reduce

def diff_x(input, r):
    assert input.dim() == 4

    left   = input[:, :,         r:2 * r + 1]
    middle = input[:, :, 2 * r + 1:         ] - input[:, :,           :-2 * r - 1]
    right  = input[:, :,        -1:         ] - input[:, :, -2 * r - 1:    -r - 1]

    output = torch.cat([left, middle, right], dim=2)

    return output

def diff_y(input, r):
    assert input.dim() == 4

    left   = input[:, :, :,         r:2 * r + 1]
    middle = input[:, :, :, 2 * r + 1:         ] - input[:, :, :,           :-2 * r - 1]
    right  = input[:, :, :,        -1:         ] - input[:, :, :, -2 * r - 1:    -r - 1]

    output = torch.cat([left, middle, right], dim=3)
    return output

#
class BoxFilter(nn.Module):
    def __init__(self, r):
        super(BoxFilter, self).__init__()
        self.r = r

    def forward(self, x):
        assert x.dim() == 4

        return diff_y(diff_x(x.cumsum(dim=2), self.r).cumsum(dim=3), self.r)

class GuidedFilter(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(GuidedFilter, self).__init__()

        self.r = r
        self.eps = eps
        self.boxfilter = kornia.filters.BoxBlur((2,2))


    def forward(self, x, y):
        n_x, c_x, h_x, w_x = x.size()
        n_y, c_y, h_y, w_y = y.size()

        assert n_x == n_y
        assert c_x == 1 or c_x == c_y
        assert h_x == h_y and w_x == w_y
        assert h_x > 2 * self.r + 1 and w_x > 2 * self.r + 1

        # N
        N = self.boxfilter(Variable(x.data.new().resize_((1, 1, h_x, w_x)).fill_(1.0)))

        # mean_x
        mean_x = self.boxfilter(x) / N
        # mean_y
        mean_y = self.boxfilter(y) / N

        # cov_xy
        cov_xy = self.boxfilter(x * y) / N - mean_x * mean_y
        # var_x
        var_x = self.boxfilter(x * x) / N - mean_x * mean_x

        # A
        A = cov_xy / (var_x + self.eps)
        # b
        b = mean_y - A * mean_x

        # mean_A; mean_b
        mean_A = self.boxfilter(A) / N
        mean_b = self.boxfilter(b) / N

        return (mean_A * x + mean_b).float()


class Grouped_Multi_Scale_Block(nn.Module):
    def __init__(self, channels):
        super(Grouped_Multi_Scale_Block, self).__init__()
        self.channels = channels
        self.conv_d1 = nn.Sequential(
            nn.Conv2d(in_channels=channels//4, out_channels=channels//4, kernel_size=3, stride=1, padding=1, dilation=1, groups=channels//4),
         
        )

        self.conv_d2 = nn.Sequential(
            nn.Conv2d(in_channels=channels//4, out_channels=channels//4, kernel_size=3, stride=1, padding=2, dilation=2, groups=channels//4),
         
        )

        self.conv_d3 = nn.Sequential(
            nn.Conv2d(in_channels=channels//4, out_channels=channels//4, kernel_size=3, stride=1, padding=3, dilation=3, groups=channels//4),
         
        )

        self.conv_d4 = nn.Sequential(
            nn.Conv2d(in_channels=channels//4, out_channels=channels//4, kernel_size=3, stride=1, padding=4, dilation=4, groups=channels//4),
            
        )
        self.relu = nn.ReLU(True)
        self.conv1x1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, stride=1, padding=0)
        self.se1 = SELayer(channels//4, r=channels//4)
        self.se2 = SELayer(channels//4, r=channels//4)
        self.se3 = SELayer(channels//4, r=channels//4)
        self.se4 = SELayer(channels//4, r=channels//4)
        self.se = SELayer(channels, r=channels//4)
    def forward(self, x):
        c = self.channels
        _4 = c // 4
        x1 = x[:, 0:_4,:,:]
        x2 = x[:,_4:(2*_4),:,:]
        x3 = x[:,(2*_4):(3*_4),:,:]
        x4 = x[:,(3*_4):c,:,:]
        x1 = self.conv_d1(x1)
        x2 = self.conv_d2(x2)
        x3 = self.conv_d3(x3)
        x4 = self.conv_d4(x4)
        x1 = self.se1(x1)
        x2 = self.se2(x2)
        x3 = self.se3(x3)
        x4 = self.se4(x4)
        # print(x1.shape, x2.shape, x3.shape, x4.shape)
        x_c = torch.cat([x1,x2,x3,x4],-3)
        x_c = self.relu(x_c)
        x_c = self.conv1x1(x_c)
        result = self.se(x_c)
        result += x
        return result

class Residual_Block(nn.Module):
    def __init__(self, channels):
        super(Residual_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        # self.ca = ChannelAttention(channels)
    def forward(self, x):
        fea = self.relu(self.bn1(self.conv1(x)))
        # part = self.ca(x)
        fea = self.relu(self.bn2(self.conv2(fea)))
        result = fea + x
        # result = result * part
        # result = self.conv1(x)
        return result
class Mobile_Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, stride=1):
        super(Mobile_Block, self).__init__()
        self.conv1 = nn.Conv2d\
            (in_planes, in_planes, kernel_size=3, stride=stride, 
             padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d\
            (in_planes, in_planes, kernel_size=1, 
            stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out



class SELayer(nn.Module):
    def __init__(self, channel, r=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c , _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Fscale
        y = torch.mul(x, y)
        return y

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


#CBAM 结构代码
#通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
 
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
 
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


# class SAU(nn.Module): 
#     def __init__(self, nFeat, conv=nn.Conv2d):
#         super(SAU, self).__init__()
#         self.c1 = conv(nFeat, nFeat//4, kernel_size=1, padding=0, bias=True)
#         self.c2 = conv(nFeat, nFeat//4, kernel_size=3, padding=1, bias=True)
#         self.c3 = conv(nFeat, nFeat//4, kernel_size=5, padding=2, bias=True)
#         self.c4 = conv(nFeat, nFeat//4, kernel_size=7, padding=3, bias=True)
#         self.down_channel_c = conv(nFeat * 2, nFeat, kernel_size=1, padding=0, bias=True)
#         self.down_channel = conv(nFeat, nFeat, kernel_size=1, padding=0, bias=True)
       
#     def forward(self, high, low):# pan mul
        
#         concat = torch.cat([high, low], -3)
      
#         concat = self.down_channel_c(concat)
#         c1 = self.c1(concat)
#         c2 = self.c2(concat)
#         c3 = self.c3(concat)
#         c4 = self.c4(concat)
#         concat = torch.cat([c1,c2,c3,c4], -3)
#         output = self.down_channel(concat) + high
#         # output = self.m(high, low) * high
#         # output = self.m(high, low)
#         return output
class SAU(nn.Module): 
    def __init__(self, nFeat, conv=nn.Conv2d):
        super(SAU, self).__init__()
        
        self.down_channel = conv(nFeat*3, nFeat, kernel_size=1, padding=0, bias=True)
       
    def forward(self, high, low):# pan mul
        
        concat = torch.cat([high, low], -3)
        add = torch.add(high, low)
        concat = torch.cat([add, concat], -3)  
        output = self.down_channel(concat)

        return output

class GFU(nn.Module): 
    def __init__(self, nFeat, conv=nn.Conv2d):
        super(GFU, self).__init__()
        
        self.m = GuidedFilter(r=2, eps=1e-2)
        self.down_channel = conv(nFeat*2, nFeat, kernel_size=3, padding=1)
    def forward(self, x, y):# pan mul
        
        # concat = torch.cat([low, high], -3)
        # mul = torch.add(low, high)
        # concat = torch.cat([concat, mul], -3)

        # output = self.down_channel(concat)
        output = self.m(x, y)
        output = torch.cat([x, output], -3)
        output = self.down_channel(output)
        return output

class A2B_RCAN1(nn.Module): # 2020 0722
    def __init__(self, args, block_num=2, scale=4, nFeat=64, Blocks=Residual_Block, conv=nn.Conv2d, part="all"):
        super(A2B_RCAN1, self).__init__()
        self.args = args
        self.part = part
        self.block_num = block_num
        self.upsample1 = Upsampler(conv=default_conv, scale=2, n_feats=nFeat, bn=False, act='relu') # act 使用激活函数relu bn改为true
        self.upsample2 = Upsampler(conv=default_conv, scale=2, n_feats=nFeat, bn=False, act='relu') # act 使用激活函数relu bn改为true
        self.mul_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=args.mul_channel, out_channels=nFeat, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2), 
        ) 
        if self.part == 'TEST' or self.part == 'SR_PLB' or self.part == 'SR_PLB_PHB' or self.part == 'SR_PLB_PHB_AEL' or self.part == 'all':
            self.sau_l1 = SAU(nFeat)
            self.sau_l2 = SAU(nFeat)
      
      
        
        self.out = nn.Sequential(
            nn.Conv2d(nFeat, args.mul_channel, kernel_size=1, padding=0),
      
        ) 

        if self.part == 'TEST' or self.part == 'SR_PHB' or self.part == 'SR_PLB_PHB' or self.part == 'SR_PLB_PHB_AEH' or self.part == 'all':
            self.pan_hp_conv_1 = nn.Sequential(
                nn.Conv2d(in_channels=args.pan_channel, out_channels=nFeat, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
            ) 
            if self.block_num != 0:
                self.pan_hp_res_1 = nn.Sequential(
                    self.make_layer(Blocks, block_num, nFeat),
                ) 
            self.pan_hp_conv_2 = nn.Sequential(
                nn.Conv2d(in_channels=nFeat, out_channels=nFeat, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2),
            )
            if self.block_num != 0:
                self.pan_hp_res_2 = nn.Sequential(
                    self.make_layer(Blocks, block_num, nFeat),
                ) 
            self.pan_hp_conv_3 = nn.Sequential(
                nn.Conv2d(in_channels=nFeat, out_channels=nFeat, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2),
            )
            if self.block_num != 0:
                self.pan_hp_res_3 = nn.Sequential(
                    self.make_layer(Blocks, block_num, nFeat),
                ) 
        if self.part == 'TEST' or self.part == 'SR_PLB' or self.part == 'SR_PLB_PHB' or self.part == 'SR_PLB_PHB_AEL' or self.part == 'all':
            self.pan_lp_conv_1 = nn.Sequential(
                nn.Conv2d(in_channels=args.pan_channel, out_channels=nFeat, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
            )
            if self.block_num != 0:
                self.pan_lp_res_1 = nn.Sequential(
                    self.make_layer(Blocks, block_num, nFeat),
                ) 
            self.pan_lp_conv_2 = nn.Sequential(
                nn.Conv2d(in_channels=nFeat, out_channels=nFeat, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2),
            )
            if self.block_num != 0:
                self.pan_lp_res_2 = nn.Sequential(
                    self.make_layer(Blocks, block_num, nFeat),
                ) 
            self.pan_lp_conv_3 = nn.Sequential(
                nn.Conv2d(in_channels=nFeat, out_channels=nFeat, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2),
            )
            if self.block_num != 0:
                self.pan_lp_res_3 = nn.Sequential(
                    self.make_layer(Blocks, block_num, nFeat),
                ) 
        if self.block_num != 0:
            self.perup1 = self.make_layer(Blocks, block_num, nFeat)
            self.perup2 = self.make_layer(Blocks, block_num, nFeat)
        
   

      
        if self.part == 'SR_PLB_PHB_AEL' or self.part == 'SR_PLB_PHB_AEH' or self.part == 'all': 
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

        
        if self.part == 'all':
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
        elif self.part == 'SR_PLB_PHB_AEL' or self.part == 'SR_PLB_PHB_AEH':
            self.decode1 = nn.Sequential(
                nn.ConvTranspose2d(in_channels=nFeat*(4+1), out_channels=nFeat*3, kernel_size=2, stride=2, padding=0),
                nn.LeakyReLU(0.2),
            )

            self.decode2 = nn.Sequential(
                nn.ConvTranspose2d(in_channels=nFeat*(3+1+2), out_channels=nFeat*3, kernel_size=2, stride=2, padding=0),
                nn.LeakyReLU(0.2),
            )

            self.decode3 = nn.Sequential(
                nn.Conv2d(in_channels=nFeat*(3+1+1), out_channels=nFeat, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
            )
        if self.part == 'TEST':
            self.perup3 = self.make_layer(Blocks, block_num, nFeat)

        self.bicubic = networks.bicubic()

    

       

    def make_layer(self, block, num_of_layer, channels):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(channels))
        return nn.Sequential(*layers)

    def forward(self, x, y):
        x_up = self.bicubic(x, scale=self.args.scale)#torch.nn.functional.upsample(x, scale_factor=cfg.scale, mode='bicubic')
        
        x = self.mul_conv_1(x)
        if self.part != 'SR':
            y_lp = kornia.filters.GaussianBlur2d((11,11),(1,1))(y)
            y_hp = y - y_lp

        if self.part == 'TEST' or self.part == 'SR_PHB' or self.part == 'SR_PLB_PHB' or self.part == 'SR_PLB_PHB_AEH' or self.part == 'all':
            y_1_hp = self.pan_hp_conv_1(y_hp)
            if self.block_num != 0:
                y_1_hp = self.pan_hp_res_1(y_1_hp) + y_1_hp
    
            y_2_hp = self.pan_hp_conv_2(y_1_hp)
            if self.block_num != 0:
                y_2_hp = self.pan_hp_res_2(y_2_hp) + y_2_hp
        
            y_3_hp = self.pan_hp_conv_3(y_2_hp)
            if self.block_num != 0:
                y_3_hp = self.pan_hp_res_3(y_3_hp) + y_3_hp

        if self.part == 'TEST' or self.part == 'SR_PLB' or self.part == 'SR_PLB_PHB' or self.part == 'SR_PLB_PHB_AEL' or self.part == 'all':
            y_1_lp = self.pan_lp_conv_1(y_lp)
            if self.block_num != 0:
                y_1_lp = self.pan_lp_res_1(y_1_lp) + y_1_lp

            y_2_lp = self.pan_lp_conv_2(y_1_lp)
            if self.block_num != 0:
                y_2_lp = self.pan_lp_res_2(y_2_lp) + y_2_lp
    
            y_3_lp = self.pan_lp_conv_3(y_2_lp)
            if self.block_num != 0:
                y_3_lp = self.pan_lp_res_3(y_3_lp) + y_3_lp
        

        if self.part == 'TEST' or self.part == 'SR_PLB' or self.part == 'SR_PLB_PHB' or self.part == 'SR_PLB_PHB_AEL' or self.part == 'all':
            x = self.sau_l1(x, y_3_lp)
        if self.block_num != 0:
            x = self.perup1(x) + x
        if self.part == 'SR' or self.part == 'SR_PLB' or self.part == 'SR_PLB_PHB_AEL':
            x = self.upsample1(x)
        if self.part == 'TEST' or self.part == 'SR_PHB' or self.part == 'SR_PLB_PHB' or self.part == 'SR_PLB_PHB_AEH'or self.part == 'all':
            x = self.upsample1(x) + y_2_hp

        if self.part == 'TEST' or self.part == 'SR_PLB' or self.part == 'SR_PLB_PHB' or self.part == 'SR_PLB_PHB_AEL' or self.part == 'all':
            x = self.sau_l2(x, y_2_lp)
        if self.block_num != 0:
            x = self.perup2(x) + x
        if self.part == 'SR' or self.part == 'SR_PLB' or self.part == 'SR_PLB_PHB_AEL':
            x = self.upsample2(x)
        if self.part == 'TEST' or self.part == 'SR_PHB' or self.part == 'SR_PLB_PHB' or self.part == 'SR_PLB_PHB_AEH' or self.part == 'all':
            x = self.upsample2(x) + y_1_hp

        if self.part == 'SR_PLB_PHB_AEL' or self.part == 'SR_PLB_PHB_AEH' or self.part == 'all': 
            e2 = self.encode2(x)
            e3 = self.encode3(e2)
            e4 = self.encode4(e3) 
        if self.part == 'all': 
            d1 = self.decode1(torch.cat([e4, y_3_lp, y_3_hp], -3))
            d2 = self.decode2(torch.cat([d1, e3, y_2_lp, y_2_hp], -3))
            d3 = self.decode3(torch.cat([d2, e2, y_1_lp, y_1_hp], -3))
        elif self.part == 'SR_PLB_PHB_AEL': 
            d1 = self.decode1(torch.cat([e4, y_3_lp], -3))
            d2 = self.decode2(torch.cat([d1, e3, y_2_lp], -3))
            d3 = self.decode3(torch.cat([d2, e2, y_1_lp], -3))
        elif self.part == 'SR_PLB_PHB_AEH': 
            d1 = self.decode1(torch.cat([e4, y_3_hp], -3))
            d2 = self.decode2(torch.cat([d1, e3, y_2_hp], -3))
            d3 = self.decode3(torch.cat([d2, e2, y_1_hp], -3))    
        if self.part == 'TEST':
            x = self.perup3(x) + x

        if self.part == 'TEST' or self.part == 'SR' or self.part == 'SR_PLB' or self.part == 'SR_PHB' or self.part == 'SR_PLB_PHB':
            x = self.out(x) + x_up 
        else:
            x = self.out(d3) + x_up 
        
        return x




      




class OursModel(BaseModel):
    
    def initialize(self, args):
        BaseModel.initialize(self, args)
        self.args = args
        self.save_dir = os.path.join(args.checkpoints_dir, args.model+args.model_sub+args.model_loss) # 定义checkpoints路径
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print("Create file path: ", self.save_dir)

        if args.isUnlabel:
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
        if args.model_sub == 'SR':
            self.netG = networks.init_net(SR()).cuda()
        elif args.model_sub == 'SR_PLB':
            self.netG = networks.init_net(SR_PLB()).cuda()
        elif args.model_sub == 'SR_PHB':
            self.netG = networks.init_net(SR_PHB()).cuda()
        elif args.model_sub == 'SR_PLB_PHB':
            self.netG = networks.init_net(SR_PLB_PHB()).cuda()

        elif args.model_sub=='F16B2':
            self.netG = networks.init_net(A2B_RCAN1(nFeat=16, block_num=2,args=args)).cuda()
        elif args.model_sub=='F16B4':
            self.netG = networks.init_net(A2B_RCAN1(nFeat=16, block_num=4,args=args)).cuda()
        elif args.model_sub=='F16B6':
            self.netG = networks.init_net(A2B_RCAN1(nFeat=16, block_num=6,args=args)).cuda()
        elif args.model_sub=='F16B8':
            self.netG = networks.init_net(A2B_RCAN1(nFeat=16, block_num=8,args=args)).cuda()
        elif args.model_sub=='F64B3':
            self.netG = networks.init_net(A2B_RCAN1(nFeat=64, block_num=3,args=args)).cuda()
        elif args.model_sub=='F64B2':
            self.netG = networks.init_net(A2B_RCAN1(nFeat=64, block_num=2,args=args)).cuda()
        elif args.model_sub=='F64B1':
            self.netG = networks.init_net(A2B_RCAN1(nFeat=64, block_num=1,args=args)).cuda()
        elif args.model_sub=='F64B0':
            self.netG = networks.init_net(A2B_RCAN1(nFeat=64, block_num=0,args=args)).cuda()
        elif args.model_sub=='F64B4':
            self.netG = networks.init_net(A2B_RCAN1(nFeat=64, block_num=4,args=args)).cuda()
        elif args.model_sub=='F64B2SR_PLB_PHB_AEL':
            self.netG = networks.init_net(A2B_RCAN1(nFeat=64, block_num=4, part='SR_PLB_PHB_AEL', args=args)).cuda()
        elif args.model_sub=='F64B2SR_PLB_PHB_AEH':
            self.netG = networks.init_net(A2B_RCAN1(nFeat=64, block_num=4, part='SR_PLB_PHB_AEH', args=args)).cuda()
        elif args.model_sub=='F64B4TEST':
            self.netG = networks.init_net(A2B_RCAN1(nFeat=64, block_num=4, part='TEST', args=args)).cuda()
        elif args.model_sub=='F64B6':
            self.netG = networks.init_net(A2B_RCAN1(nFeat=64, block_num=6,args=args)).cuda()
        elif args.model_sub=='F64B8':
            self.netG = networks.init_net(A2B_RCAN1(nFeat=64, block_num=8,args=args)).cuda()
        elif args.model_sub=='F32B2':
            self.netG = networks.init_net(A2B_RCAN1(nFeat=32, block_num=2,args=args)).cuda()
        elif args.model_sub=='F32B4':
            self.netG = networks.init_net(A2B_RCAN1(nFeat=32, block_num=4,args=args)).cuda()
        elif args.model_sub=='F32B6':
            self.netG = networks.init_net(A2B_RCAN1(nFeat=32, block_num=6,args=args)).cuda()
        elif args.model_sub=='F32B8':
            self.netG = networks.init_net(A2B_RCAN1(nFeat=32, block_num=8,args=args)).cuda()
        elif args.model_sub=='F48B2':
            self.netG = networks.init_net(A2B_RCAN1(nFeat=48, block_num=2,args=args)).cuda()
        elif args.model_sub=='F48B4':
            self.netG = networks.init_net(A2B_RCAN1(nFeat=48, block_num=4,args=args)).cuda()
        elif args.model_sub=='F48B6':
            self.netG = networks.init_net(A2B_RCAN1(nFeat=48, block_num=6,args=args)).cuda()
        elif args.model_sub=='F48B8':
            self.netG = networks.init_net(A2B_RCAN1(nFeat=48, block_num=8,args=args)).cuda()
        else:
            self.netG = networks.init_net(A2B_RCAN1()).cuda()
      

        if self.isTrain:
            
            
            # define loss functions
            
            if args.isUnlabel:
                self.criterionL1 = networks.OursLoss(scale=args.scale, device=self.device)#
            else:
                if args.model_sub=='DIP':
                    # self.criterionL1 = networks.S3Loss(scale=cfg.scale, device=self.device)
                    self.criterionL1 = networks.DIPLoss(device=self.device, args=args)
                elif args.model_loss=='SSIM':
                    # self.criterionL1 = networks.S3Loss(scale=cfg.scale, device=self.device)
                    self.criterionL1 = networks.DIPLoss(device=self.device, args=args)
                elif args.model_loss=='L1':
                    self.criterionL1 = networks.L1Loss(device=self.device)#torch.nn.L1Loss()
                else:
                    self.criterionL1 = networks.OursLoss2(scale=args.scale, device=self.device)#
            # initialize optimizers
    
            if args.optim_type=='adam':
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=args.lr, betas=(args.beta, 0.999), weight_decay=args.weight_decay)
            elif args.optim_type=='sgd':
                self.optimizer_G = torch.optim.SGD(self.netG.parameters(),
                                                    lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
      
        

    def set_input(self, input_dict):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        if self.args.isUnlabel:
            self.real_A_1 = input_dict['A_1'].to(self.device)  # mul
            self.real_A_2 = input_dict['A_2'].to(self.device)  # pan
        else:
            self.real_A_1 = input_dict['A_1'].to(self.device)  # mul
            self.real_A_2 = input_dict['A_2'].to(self.device)  # pan
            self.real_B = input_dict['B'].to(self.device) # fus
          

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.args.isUnlabel:
            self.fake_B, self.fake_pan = self.netG(self.real_A_1, self.real_A_2) 
        else:
            self.fake_B = self.netG(self.real_A_1, self.real_A_2) 

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        #
       
        if self.args.isUnlabel:
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