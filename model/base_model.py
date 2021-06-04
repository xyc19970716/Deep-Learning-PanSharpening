import os
import itertools
from collections import OrderedDict
import torch
from abc import ABC, abstractmethod
from . import networks
import config as cfg
import numpy as np
import math
from tqdm import tqdm
import cv2

class BaseModel(ABC):

    def setup(self):
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, self.args) for optimizer in self.optimizers]
        if not self.isTrain or self.args.continue_train:
            self.load_networks(self.args.which_epoch)
        # self.print_networks(opt.verbose)
        self.print_named_networks(self.args.print_net_in_detail)

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    def update_learning_rate(self, decay_factor=0.1):
        """Update learning rates for all the networks; called at the end of every epoch"""
        
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr'] 

        infos = []
        info = {}
        info['old_lr'] = old_lr
        info['new_lr'] = lr
        info['name'] = 'G'
        infos.append(info)
        return infos

    # save models to the disk
    def save_networks(self, which_epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (which_epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)
                torch.save(net.state_dict(), save_path)


    # load models from the disk
    def load_networks(self, which_epoch):
        for name in self.model_names:
            if isinstance(name, str):
                if which_epoch==-1: # if -1, load best
                    load_filename = 'best_net_%s.pth' % (name)
                else:
                    load_filename = '%s_net_%s.pth' % (which_epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # now this verison is 1.1.0 (*^_^*)
                # # patch InstanceNorm checkpoints prior to 0.4
                # for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                #     self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    # print network information
    def print_networks(self, verbose):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def print_named_networks(self, verbose):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for name, param in net.named_parameters():
                    print(name, param.numel())
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def initialize(self, args):
        self.isTrain = args.isTrain # 是否训练
        self.gpu_ids = args.gpu_ids # 显卡号
        self.device = torch.device('cuda:{}'.format(args.gpu_ids)) if int(args.gpu_ids) > -1 else torch.device('cpu') # 设置设备
       
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.image_paths = []
        self.optimizers = []
        self.metric = 0
        self.args = args

        
        
    @abstractmethod
    def set_input(self, input_dict):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        pass
          
    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass 


    @abstractmethod
    def optimize_parameters(self):
        pass

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def train(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()

    def predict(self, X, Y):
        
        M, N, COUNT = Y.shape
        m, n, count = X.shape
        XX = X
        self.eval()
        if self.args.model=='CycleMP':
            net = self.netG_A
        else:
            net = self.netG

        mul_channel = self.args.mul_channel
        pan_channel = self.args.pan_channel
        data_range = self.args.data_range
        
        """eval"""
        # print('evaling...')
        torch.cuda.empty_cache()
        with torch.no_grad():

            # result = np.zeros((count, M, N))  
            # array_data_batch = np.transpose(X, (2, 0, 1))
            # X = array_data_batch
            # X = X.reshape(1, mul_channel, m, n)
            # X = torch.Tensor(X).cuda()
                
            # array_data2_batch = np.transpose(Y, (2, 0, 1))
            # Y = array_data2_batch
            # Y = Y.reshape(1, pan_channel, M, N)
            # Y = torch.Tensor(Y).cuda()

            # # print(X, Y)
            # if cfg.isUnlabel:
            #     extract_result, _ = net(X, Y)
            # else:
            #     extract_result = net(X, Y)
            # extract_result = extract_result.cpu().detach().numpy()
            # #adjustment
            # extract_result[extract_result<0]=0
            # extract_result[extract_result>1]=1

            # extract_result =  extract_result * data_range
            # b, c, h, w = extract_result.shape
            # # print(extract_result)
            
            # r = extract_result.reshape(c, h, w)

            # for l in range(c): 
            #     result[l] = r[l]
            lrhs = X
            hrms = Y
            clip_size = 128#128
            pad_num = clip_size // 4
            scales = 4
            batch_idx = 0
            batch_iter = 0
            batch_size = 1#4
            loc = []

            rows_iters = math.ceil(m / (clip_size-(pad_num*2))) #??
            cols_iters = math.ceil(n / (clip_size-(pad_num*2))) #??
           
            lrhs = np.pad(lrhs, ((pad_num,pad_num), (pad_num,pad_num),(0,0)), 'edge')#'constant', constant_values=0
            hrms = np.pad(hrms, ((pad_num * scales,pad_num * scales), (pad_num *scales,pad_num*scales), (0,0)), 'edge')
            padded_h, padded_w, padded_c = lrhs.shape
            padded_h2, padded_w2, padded_c2 = hrms.shape

            result = np.zeros((count, M, N))  

            if rows_iters == 1 or cols_iters==1:
                array_data_batch = np.transpose(X, (2, 0, 1))
                X = array_data_batch
                X = X.reshape(1, mul_channel, m, n)
                X = torch.Tensor(X).cuda()
                    
                array_data2_batch = np.transpose(Y, (2, 0, 1))
                Y = array_data2_batch
                Y = Y.reshape(1, pan_channel, M, N)
                Y = torch.Tensor(Y).cuda()

               
                extract_result = net(X, Y)
                extract_result = extract_result.cpu().detach().numpy()
                #adjustment
                extract_result[extract_result<0]=0
                extract_result[extract_result>1]=1

                extract_result =  extract_result * data_range
                b, c, h, w = extract_result.shape
                # print(extract_result)
                
                r = extract_result.reshape(c, h, w)

                for l in range(c): 
                    result[l] = r[l]
            else:
                for i in tqdm(range(rows_iters)): #??
                    for j in range(cols_iters): #??
                        h_s = i * (clip_size - pad_num * 2)
                        h_e = i * (clip_size - pad_num * 2) + clip_size
                        w_s = j * (clip_size - pad_num * 2)
                        w_e = j * (clip_size - pad_num * 2) + clip_size
                        
                        r_h = i * (clip_size - pad_num * 2) 
                        r_w = j * (clip_size - pad_num * 2) 

                        if i == rows_iters - 1:
                            h_s = padded_h2 - clip_size * scales
                            h_e = padded_h2
                            r_h = M - (clip_size*scales - (pad_num*scales) * 2) 
                            h_s = h_s // scales
                            h_e = h_e // scales
                            r_h = r_h // scales
                        if j == cols_iters - 1:
                            w_s = padded_w2 - clip_size *scales
                            w_e = padded_w2
                            r_w = N  - (clip_size*scales - (pad_num*scales) * 2)
                            w_s = w_s // scales
                            w_e = w_e // scales
                            r_w = r_w // scales
                            
                        # print('行列',h_s, w_s, '写入',r_h, r_w)
                        array_data_batch = lrhs[h_s:h_e, w_s:w_e, :]
                        # print(array_data_batch.shape)
                        array_data_batch = np.transpose(array_data_batch, (2, 0, 1))
                        X = array_data_batch
                        X = X.reshape(1, mul_channel, clip_size, clip_size)
                        X = torch.Tensor(X).cuda()
                        if batch_idx == 0:
                            x_batch = X
                        else:
                            x_batch = torch.cat([x_batch, X], 0)


                        array_data2_batch = hrms[h_s*scales:h_s*scales+clip_size*scales, w_s*scales:w_s*scales+clip_size*scales, :]
                        array_data2_batch = np.transpose(array_data2_batch, (2, 0, 1))
                        Y = array_data2_batch
                        Y = Y.reshape(1, pan_channel, clip_size * scales, clip_size * scales)
                        Y = torch.Tensor(Y).cuda()
                        if batch_idx == 0:
                            y_batch = Y
                        else:
                            y_batch = torch.cat([y_batch, Y], 0)

                        loc_s = []
                        loc_s.append(r_w)
                        loc_s.append(r_h)
                        loc.append(loc_s)
                        batch_idx +=1

                        if batch_idx == batch_size:

                            extract_result = net(x_batch, y_batch)
                            extract_result = extract_result.cpu().detach().numpy()
                            #adjustment
                            extract_result[extract_result<0]=0
                            extract_result[extract_result>1]=1

                            extract_result =  extract_result * data_range
                            b, c, h, w = extract_result.shape
                            
                            for b_s in range(b):
                                r = extract_result[b_s].reshape(c, h, w)

                                for l in range(c): 
                                    hh = h - (pad_num*scales)*2
                                    ww = w - (pad_num*scales)*2
                                    # print(loc[b_s][1], loc[b_s][0])
                                    result[l][loc[b_s][1] * scales:loc[b_s][1] * scales+hh, loc[b_s][0] * scales:loc[b_s][0] * scales+ww] = r[l][(pad_num*scales):h-(pad_num*scales), (pad_num*scales):w-(pad_num*scales)]

                            batch_idx = 0
                            loc = []
                            batch_iter += 1

        result = np.uint16(result.transpose((1,2,0)))
        torch.cuda.empty_cache()

        return result

