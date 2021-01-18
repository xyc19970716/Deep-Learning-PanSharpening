
import os
import itertools
from collections import OrderedDict
import torch.nn as nn
import torch
from .base_model import BaseModel
from . import networks
import config as cfg
import numpy as np
import math
import kornia
try:    
    from apex import amp
    fp16_mode = False#True
except ImportError:
    fp16_mode = False
    print("Please install apex from https://www.github.com/nvidia/apex to speed up this model.")


        
class PNN_model(nn.Module):
    def __init__(self):
        super(PNN_model, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=cfg.mul_channel+cfg.pan_channel, out_channels=64, kernel_size=9, stride=1, padding=4)
        self.conv_2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv_3 = nn.Conv2d(in_channels=32, out_channels=cfg.mul_channel, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bicubic = networks.bicubic()
     
    def forward(self, x, y):
        x = self.bicubic(x, scale=cfg.scale)#torch.nn.functional.interpolate(x, scale_factor=cfg.scale, mode='bicubic', align_corners=True)#
        in_put = torch.cat([x,y], -3)
        # in_put = torch.cat([x,y], -3)
        fea = self.relu(self.conv_1(in_put))  
        fea =  self.relu(self.conv_2(fea))
        out = self.conv_3(fea)

        return out
        
class PNNModel(BaseModel):

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
        self.netG = networks.init_net(PNN_model()).cuda()
        
        if self.isTrain:
            
            # define loss functions
            # 
            # self.criterionL1 = networks.UNPANLoss(scale=cfg.scale, device=self.device)#
            if cfg.isUnlabel:
                self.criterionL1 = networks.OursLoss(scale=cfg.scale, device=self.device)#
            else:
            
                self.criterionL1 = torch.nn.MSELoss()#networks.PanLoss(scale=cfg.scale, device=self.device)#
            # initialize optimizers
            # self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
            #                                     lr=cfg.lr, betas=(cfg.beta, 0.999))
            conv_3_params = list(map(id, self.netG.conv_3.parameters()))
            base_params = filter(lambda p: id(p) not in conv_3_params,
                                self.netG.parameters())
            if cfg.optim_type=='adam':
                self.optimizer_G = torch.optim.Adam([{'params': base_params},
                        {'params': self.netG.conv_3.parameters(), 'lr': cfg.lr * 0.1}], lr=cfg.lr, betas=(cfg.beta, 0.999), weight_decay=cfg.weight_decay)
            elif cfg.optim_type=='sgd':
                self.optimizer_G = torch.optim.SGD([{'params': base_params},
                        {'params': self.netG.conv_3.parameters(), 'lr': cfg.lr * 0.1}], lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
           
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
          
            self.loss_G = self.criterionL1(self.fake_B, self.real_B)
       
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

