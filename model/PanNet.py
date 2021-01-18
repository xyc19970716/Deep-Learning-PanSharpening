
import os
import itertools
from collections import OrderedDict
import torch.nn as nn
import torch
from .base_model import BaseModel
from . import networks
import config as cfg
import numpy as np
import cv2
import kornia

    
import torch
import config as cfg
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
       

class PanNet_model(nn.Module):
    def __init__(self):
        super(PanNet_model, self).__init__()
        self.layer_0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=cfg.mul_channel, out_channels=cfg.mul_channel, kernel_size=8, stride=4, padding=2, output_padding=0)
        )
        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=cfg.pan_channel+cfg.mul_channel, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.layer_2 = nn.Sequential(
            self.make_layer(Residual_Block, 4, 32),
            nn.Conv2d(in_channels=32, out_channels=cfg.mul_channel, kernel_size=3, stride=1, padding=1)
        )

        self.bicubic = networks.bicubic()

    def make_layer(self, block, num_of_layer, channels):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(channels))
        return nn.Sequential(*layers)

    def forward(self, x, y):
        lr_up = self.bicubic(x, scale=cfg.scale)#torch.nn.functional.interpolate(lr, scale_factor=cfg.scale, mode='bicubic')
        lr_hp = x - kornia.filters.BoxBlur((5, 5))(x)
        pan_hp = y - kornia.filters.BoxBlur((5, 5))(y)
        lr_u_hp = self.layer_0(lr_hp)#self.bicubic(lr_hp, scale=cfg.scale)#
        ms = torch.cat([pan_hp, lr_u_hp], dim=1)
        fea = self.layer_1(ms)
        output = self.layer_2(fea) + lr_up

        return output


    
class PanNetModel(BaseModel):

    def initialize(self):
        BaseModel.initialize(self)
        self.save_dir = os.path.join(cfg.checkpoints_dir, cfg.model) # 定义checkpoints路径
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
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG = networks.init_net(PanNet_model()).cuda()
        

        if self.isTrain:
            
            
            # define loss functions
            if cfg.isUnlabel:
                self.criterionL1 = networks.OursLoss(scale=cfg.scale, device=self.device)#
            else:
             
                self.criterionL1 = torch.nn.MSELoss()#networks.PanLoss(scale=cfg.scale, device=self.device)#
          
            
            # initialize optimizers
            if cfg.optim_type == 'adam':
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=cfg.lr, betas=(cfg.beta, 0.999), weight_decay=cfg.weight_decay)
            elif cfg.optim_type == 'sgd':
                self.optimizer_G = torch.optim.SGD(self.netG.parameters(), lr=cfg.lr, momentum=cfg.momentum)

            
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
        if cfg.isUnlabel:
            self.loss_G = self.criterionL1(self.real_A_1, self.real_A_2,self.fake_B, self.fake_pan)
        else:
        
            self.loss_G = self.criterionL1(self.fake_B, self.real_B)

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights


