
import os
import itertools
from collections import OrderedDict
import torch.nn as nn
import torch
from .base_model import BaseModel
from . import networks
import numpy as np
import math



class MSDCNN_model(nn.Module):
    def __init__(self, args):
        super(MSDCNN_model, self).__init__()
        self.args = args
        self.shallow_conv_1 = nn.Conv2d(in_channels=args.mul_channel+args.pan_channel, out_channels=64, kernel_size=9, stride=1, padding=4)
        self.shallow_conv_2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.shallow_conv_3 = nn.Conv2d(in_channels=32, out_channels=args.mul_channel, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.deep_conv_1 = nn.Conv2d(in_channels=args.mul_channel+args.pan_channel, out_channels=60, kernel_size=7, stride=1, padding=3)
        self.deep_conv_1_sacle_1 = nn.Conv2d(in_channels=60, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.deep_conv_1_sacle_2 = nn.Conv2d(in_channels=60, out_channels=20, kernel_size=5, stride=1, padding=2)
        self.deep_conv_1_sacle_3 = nn.Conv2d(in_channels=60, out_channels=20, kernel_size=7, stride=1, padding=3)
        self.deep_conv_2 = nn.Conv2d(in_channels=60, out_channels=30, kernel_size=3, stride=1, padding=1)
        self.deep_conv_2_sacle_1 = nn.Conv2d(in_channels=30, out_channels=10, kernel_size=3, stride=1, padding=1)
        self.deep_conv_2_sacle_2 = nn.Conv2d(in_channels=30, out_channels=10, kernel_size=5, stride=1, padding=2)
        self.deep_conv_2_sacle_3 = nn.Conv2d(in_channels=30, out_channels=10, kernel_size=7, stride=1, padding=3)
        self.deep_conv_3 = nn.Conv2d(in_channels=30, out_channels=args.mul_channel, kernel_size=5, stride=1, padding=2)
        self.bicubic = networks.bicubic()


    def forward(self, x, y):
        x = self.bicubic(x, scale=self.args.scale)#x = torch.nn.functional.interpolate(x, scale_factor=cfg.scale, mode='bicubic')
        in_put = torch.cat([x,y], -3)
   
        shallow_fea = self.relu(self.shallow_conv_1(in_put))  
        shallow_fea =  self.relu(self.shallow_conv_2(shallow_fea))
        shallow_out = self.shallow_conv_3(shallow_fea)

        deep_fea = self.relu(self.deep_conv_1(in_put))
        deep_fea_scale1=self.relu(self.deep_conv_1_sacle_1(deep_fea))
        deep_fea_scale2=self.relu(self.deep_conv_1_sacle_2(deep_fea))
        deep_fea_scale3=self.relu(self.deep_conv_1_sacle_3(deep_fea))
        deep_fea_scale = torch.cat([deep_fea_scale1, deep_fea_scale2, deep_fea_scale3], -3)
        deep_fea_1 = torch.add(deep_fea, deep_fea_scale)
        deep_fea_2 = self.relu(self.deep_conv_2(deep_fea_1))
        deep_fea_2_scale1=self.relu(self.deep_conv_2_sacle_1(deep_fea_2))
        deep_fea_2_scale2=self.relu(self.deep_conv_2_sacle_2(deep_fea_2))
        deep_fea_2_scale3=self.relu(self.deep_conv_2_sacle_3(deep_fea_2))
        deep_fea_2_scale = torch.cat([deep_fea_2_scale1, deep_fea_2_scale2, deep_fea_2_scale3], -3)
        deep_fea_3 = torch.add(deep_fea_2, deep_fea_2_scale)
        deep_out = self.deep_conv_3(deep_fea_3)

        out = deep_out + shallow_out

        return out
        
class MSDCNNModel(BaseModel):

    def initialize(self, args):
        self.args = args
        BaseModel.initialize(self, args)
        self.save_dir = os.path.join(args.checkpoints_dir, args.model) # 定义checkpoints路径
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
        self.netG = networks.init_net(MSDCNN_model(args)).cuda()
        
        if self.isTrain:
            
            # define loss functions
            if args.isUnlabel:
                self.criterionL1 = networks.OursLoss(scale=args.scale, device=self.device)#
            else:
                self.criterionL1 = torch.nn.MSELoss()#networks.PanLoss(scale=cfg.scale, device=self.device)#
          
            # initialize optimizers
            
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=args.lr, betas=(args.beta, 0.999), weight_decay=args.weight_decay)
            # self.optimizer_G = torch.optim.SGD([{'params': base_params},
            #         {'params': self.netG.conv_3.parameters(), 'lr': cfg.lr * 0.1}], lr=cfg.lr, momentum=cfg.beta, weight_decay=cfg.weight_decay)

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
        if self.args.isUnlabel:
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

