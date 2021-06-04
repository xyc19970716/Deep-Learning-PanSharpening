
import os
import itertools
from collections import OrderedDict
import torch.nn as nn
import torch
from .base_model import BaseModel
from . import networks

import numpy as np
import cv2
import kornia
    
import torch

class Residual_Block(nn.Module):
    def __init__(self, channels):
        super(Residual_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
       

    def forward(self, x):
        fea = self.relu(self.conv1(x))
        fea = self.conv2(fea)
        result = fea + x
        return result

class SRPPNN_model(nn.Module):
    def __init__(self, args):
        super(SRPPNN_model, self).__init__()
        self.args = args
        self.bicubic =networks.bicubic()
        self.pan_extract_1 = nn.Sequential(
            nn.Conv2d(in_channels=args.pan_channel, out_channels=64, kernel_size=3, stride=1, padding=1),
            self.make_layer(Residual_Block, 10, 64),
            nn.Conv2d(in_channels=64, out_channels=args.mul_channel, kernel_size=3, stride=1, padding=1),
        )
        self.pan_extract_2 = nn.Sequential(
            nn.Conv2d(in_channels=args.mul_channel, out_channels=64, kernel_size=3, stride=1, padding=1),
            self.make_layer(Residual_Block, 10, 64),
            nn.Conv2d(in_channels=64, out_channels=args.mul_channel, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2),
        )
        self.ms_up_1 = nn.Sequential(
            nn.Conv2d(in_channels=args.mul_channel, out_channels=args.mul_channel*4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
        )
        self.ms_up_2 = nn.Sequential(
            nn.Conv2d(in_channels=args.mul_channel, out_channels=args.mul_channel*4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
        )

        self.conv_mul_pre_p1 = nn.Conv2d(in_channels=args.mul_channel, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.img_mul_p1_layer = self.make_layer(Residual_Block, 4, 32)
        self.conv_mul_post_p1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.mul_grad_p1 = nn.Conv2d(in_channels=32, out_channels=args.mul_channel, kernel_size=3, stride=1, padding=1)
        self.mul_grad_p2 = nn.Conv2d(in_channels=32, out_channels=args.mul_channel, kernel_size=3, stride=1, padding=1)
        self.conv_pre_p1 = nn.Conv2d(in_channels=args.mul_channel+args.pan_channel, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_pre_p2 = nn.Conv2d(in_channels=args.mul_channel+args.pan_channel, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.img_p1_layer = self.make_layer(Residual_Block, 4, 32)
        self.conv_post_p1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_post_p2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.grad_p1 = nn.Conv2d(in_channels=32, out_channels=args.mul_channel, kernel_size=3, stride=1, padding=1)
        self.grad_p2 = nn.Conv2d(in_channels=32, out_channels=args.mul_channel, kernel_size=3, stride=1, padding=1)
        self.conv_mul_pre_p2 = nn.Conv2d(in_channels=args.mul_channel, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.img_mul_p2_layer = self.make_layer(Residual_Block, 4, 32)
        self.conv_mul_post_p2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.img_p2_layer = self.make_layer(Residual_Block, 4, 32)
    def make_layer(self, block, num_of_layer, channels):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(channels))
        return nn.Sequential(*layers)

    def forward(self, x, y):
        inputs_mul_up_p1 = self.bicubic(x, scale=2)
        inputs_mul_up_p2 = self.bicubic(x, scale=4)
        inputs_pan = y
        inputs_pan_blur = kornia.filters.GaussianBlur2d((11,11),(1,1))(y)
        inputs_pan_down_p1 = self.bicubic(inputs_pan_blur, scale=1/2)
        pre_inputs_mul_p1_feature = self.conv_mul_pre_p1(x)
        x = pre_inputs_mul_p1_feature
        x = self.img_mul_p1_layer(x)
        post_inputs_mul_p1_feature = self.conv_mul_post_p1(x)
        inputs_mul_p1_feature = pre_inputs_mul_p1_feature + post_inputs_mul_p1_feature
        inputs_mul_p1_feature_bic = self.bicubic(inputs_mul_p1_feature, scale=2)
        net_img_p1_sr = self.mul_grad_p1(inputs_mul_p1_feature_bic) + inputs_mul_up_p1
        inputs_p1 = torch.cat([net_img_p1_sr, inputs_pan_down_p1], -3)
        
        pre_inputs_p1_feature = self.conv_pre_p1(inputs_p1)
        x = pre_inputs_p1_feature
        x = self.img_p1_layer(x)
        post_inputs_p1_feature = self.conv_post_p1(x)
        inputs_p1_feature = pre_inputs_p1_feature+post_inputs_p1_feature

        inputs_pan_down_p1_blur = kornia.filters.GaussianBlur2d((11,11),(1,1))(inputs_pan_down_p1)
        inputs_pan_hp_p1 = inputs_pan_down_p1 - inputs_pan_down_p1_blur
        net_img_p1 = self.grad_p1(inputs_p1_feature) + inputs_mul_up_p1 + inputs_pan_hp_p1

        pre_inputs_mul_p2_feature = self.conv_mul_pre_p2(net_img_p1)
        x = pre_inputs_mul_p2_feature
        x = self.img_mul_p2_layer(x)
        post_inputs_mul_p2_feature = self.conv_mul_post_p2(x)
        inputs_mul_p2_feature = pre_inputs_mul_p2_feature+post_inputs_mul_p2_feature
        inputs_mul_p2_feature_bic = self.bicubic(inputs_mul_p2_feature, scale=2)
        net_img_p2_sr = self.mul_grad_p2(inputs_mul_p2_feature_bic) + inputs_mul_up_p2
        inputs_p2 = torch.cat([net_img_p2_sr, inputs_pan], -3)

        pre_inputs_p2_feature = self.conv_pre_p2(inputs_p2)
        x = pre_inputs_p2_feature
        x = self.img_p2_layer(x)
        post_inputs_p2_feature = self.conv_post_p2(x)
        inputs_p2_feature = pre_inputs_p2_feature+post_inputs_p2_feature

        inputs_pan_hp_p2 = inputs_pan - inputs_pan_blur
        net_img_p2 = self.grad_p2(inputs_p2_feature) + inputs_mul_up_p2 + inputs_pan_hp_p2
        return net_img_p2


    
class SRPPNNModel(BaseModel):

    def initialize(self, args):
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
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG = networks.init_net(SRPPNN_model(args)).cuda()
        

        if self.isTrain:
            
            
            # define loss functions
            if args.isUnlabel:
                self.criterionL1 = networks.OursLoss(scale=args.scale, device=self.device)#
            else:
            
                self.criterionL1 = torch.nn.MSELoss()#networks.PanLoss(scale=cfg.scale, device=self.device)#
          
            
            # initialize optimizers
            if args.optim_type == 'adam':
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=args.lr, betas=(args.beta, 0.999), weight_decay=args.weight_decay)
            elif args.optim_type == 'sgd':
                self.optimizer_G = torch.optim.SGD(self.netG.parameters(), lr=args.lr, momentum=args.momentum)

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


