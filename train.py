import math
import time
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import argparse
import numpy as np
from torch.nn import init
import torch.optim as optim
import torch
import random
from model.Ours import OursModel
from model.PNN import PNNModel
from model.PanNet import PanNetModel
from model.TFNet import TFNetModel
from model.MSDCNN import MSDCNNModel
from model.SRPPNN import SRPPNNModel

from data import PsDataset, Get_DataSet
from utils import *
import config as cfg
import metrics
import warnings

from matplotlib import pyplot as plt
import math
import gdal
from tqdm import tqdm



# 判断当前次数是否过了设定衰减，返回索引
def detectIndexInDecayIters(step, iters):
    for i in range(len(iters)):
        if step < iters[i]:
            return i
    return len(iters)-1

def get_dataset():

    data_train = PsDataset(apath=cfg.dataDir, isUnlabel=cfg.isUnlabel)#PsRamDataset(apath=cfg.dataDir, isUnlabel=cfg.isUnlabel)#LmdbDataset()#
    dataloader = torch.utils.data.DataLoader(data_train, batch_size=cfg.batchSize,
                                             drop_last=True, shuffle=True, num_workers=int(cfg.nThreads),
                                             pin_memory=True)
    return dataloader

def evalOrSaveBest(net, dataloader, best_eval_index):
    # 评估训练模型，保存效果最好模型
    step = 0
    current_eval_index = 0
    
    # 
    val_loss = 0
    val_rmse = 0
    val_psnr = 0
    net.eval()
    with torch.no_grad():
        
        for batch, (im_lr, im_hr, im_fr) in enumerate(dataloader):
            
            img_low_resolution = Variable(im_lr.cuda(), volatile=False)
            img_high_resolution = Variable(im_hr.cuda())
            img_pansherpen = Variable(im_fr.cuda())
            input_dict = {'A_1': img_low_resolution,
                        'A_2': img_high_resolution,
                        'B': img_pansherpen}
            net.set_input(input_dict)
            net.forward()
            
            fake_B = net.fake_B.cpu().detach().numpy() * cfg.data_range
            real_B = net.real_B.cpu().detach().numpy() * cfg.data_range
            
            current_batch_eval_index = metrics.get_rmse(real_B, fake_B) 

            current_eval_index += current_batch_eval_index
            
            print('Valing: {}'.format(step), 'current_cc: {}'.format(current_batch_eval_index / cfg.batchSize))
            step += 1
            
            
            
            val_loss += net.loss_G
            val_rmse += current_batch_eval_index
            val_psnr += metrics.psnr(fake_B, real_B, dynamic_range=cfg.data_range)
        

    # print(len(dataloader))
    current_eval_index = current_eval_index / len(dataloader) / cfg.batchSize
    
    print('val_cc=', current_eval_index)
    if current_eval_index < best_eval_index:
        print('better than best_cc=', best_eval_index, 'save to best')
        best_eval_index = current_eval_index
        net.save_networks('best')
    return best_eval_index, val_loss / len(dataloader), val_rmse / len(dataloader), val_psnr / len(dataloader)

def gdal_write(output_file,array_data):
    
    #判断栅格数据的数据类型0
    
    if 'int8' in array_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in array_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    c,h,w = array_data.shape
    Driver = gdal.GetDriverByName("Gtiff")
    dataset = Driver.Create(output_file,w,h,c,datatype)
    
    for i in range(c):
        band = dataset.GetRasterBand(i+1)
        band.WriteArray(array_data[i,:,:])

def train():
    log = LossLog()

    #  select network
    if cfg.model == 'ours':
        cycle_gan = OursModel()
    elif cfg.model == 'PNN':
        cycle_gan = PNNModel()
    elif cfg.model == 'PanNet':
        cycle_gan = PanNetModel()
    elif cfg.model == 'TFNet':
        cycle_gan = TFNetModel()
    elif cfg.model == 'PSGAN':
        cycle_gan = PSGANModel()
    elif cfg.model == 'MSDCNN':
        cycle_gan = MSDCNNModel()
    elif cfg.model == 'SRPPNN':
        cycle_gan = SRPPNNModel()
  

    print(cycle_gan)
    cycle_gan.initialize()
    # cycle_gan.cuda()

    cycle_gan.setup()
    
    # load data
    # dataloader, dataloader2 = get_dataset()
    dataloader = get_dataset()

    batch_iter = 0 # iterations
    lr_decay_iters_idx = 0


    cycle_gan.train()
    
    best_psnr = 999999

    # mse
    mse = nn.MSELoss()
    # 计算一代的迭代次数
    epoch_iter_nums = len(dataloader)
    # 计算总共迭代次数
    total_iter_nums = epoch_iter_nums * cfg.epochs

    # 训练集平均loss
    avg_loss = 0
    # 训练集评价rmse
    avg_rmse = 0
    # 评估集评价loss
    avg_val_loss = 0
    # 评估集平均rmse
    avg_val_rmse = 0
    # 训练集lossHistory
    loss_history = []
    # 训练集rmseHistory
    rmse_history = []
    # 评估集lossHistory
    val_loss_history = []
    # 评估集rmseHistory
    val_rmse_history = []


    # 训练集评价psnr
    avg_psnr = 0
    # 评估集平均rmse
    avg_val_psnr = 0
    # 训练集rmseHistory
    psnr_history = []
    # 评估集rmseHistory
    val_psnr_history = []

    # 迭代数据集次数轴
    epoch_history = [i+1 for i in range(cfg.epochs)]


    for epoch in range(cfg.which_epoch+1, cfg.epochs):
        iter_data_time = time.time()
     
        
        
        for batch, (im_lr, im_hr, im_fr) in enumerate(dataloader):
            iter_start_time = time.time()
            
            img_low_resolution = Variable(im_lr.cuda(), volatile=False)
            img_high_resolution = Variable(im_hr.cuda())
            img_pansherpen = Variable(im_fr.cuda())
            input_dict = {'A_1': img_low_resolution,
                        'A_2': img_high_resolution,
                        'B': img_pansherpen}

            cycle_gan.set_input(input_dict)
            cycle_gan.optimize_parameters()

            losses = cycle_gan.get_current_losses()

            # 获取loss
            for k,v in losses.items():
                if k == 'G':
                    avg_loss += v
            
            # 获取rmse
            avg_rmse += metrics.get_rmse(cycle_gan.fake_B.detach().cpu().numpy(), cycle_gan.real_B.detach().cpu().numpy())
            avg_psnr += metrics.psnr(cycle_gan.fake_B.detach().cpu().numpy()*cfg.data_range, cycle_gan.real_B.detach().cpu().numpy()*cfg.data_range, dynamic_range=cfg.data_range)
            # 打印信息
            if (batch_iter+1) % cfg.print_freq == 0: 
                t = (time.time() - iter_start_time) / cfg.batchSize
                t_data = iter_start_time - iter_data_time
                log.print_current_losses(epoch, batch, epoch_iter_nums, batch_iter, total_iter_nums, losses, t, t_data)
            

            batch_iter += 1
        
        # 
        loss_history.append(avg_loss / epoch_iter_nums)
        avg_loss = 0
        rmse_history.append(avg_rmse / epoch_iter_nums)
        avg_rmse = 0
        psnr_history.append(avg_psnr / epoch_iter_nums)
        avg_psnr = 0

        # 更新学习率
        change_infos = cycle_gan.update_learning_rate(decay_factor=cfg.lr_decay_factor)
        # 显示并记录日志
        for info in change_infos:
            log.print_change_learning_rate(epoch, info['name'], info['old_lr'], info['new_lr'])

        if cfg.isEval:
            best_psnr, avg_val_loss, avg_val_rmse, avg_val_psnr = evalOrSaveBest(net=cycle_gan, dataloader=dataloader2, best_eval_index=best_psnr)  
        else:
            best_psnr, avg_val_loss, avg_val_rmse, avg_val_psnr = 0,0,0,0
        val_loss_history.append(avg_val_loss)
        val_rmse_history.append(avg_val_rmse)
        val_psnr_history.append(avg_val_psnr)

        if (epoch+1) % cfg.save_epoch_freq == 0:
            cycle_gan.save_networks(epoch)

    print('final best =', best_psnr)
    plt.plot(epoch_history, loss_history, 'r', label = 'Training loss')
    plt.plot(epoch_history, val_loss_history, 'b', label = 'Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    path = os.path.join(cycle_gan.save_dir, 'loss.png')
    plt.savefig(path, dpi = 300)

    plt.figure()
    plt.plot(epoch_history, rmse_history, 'r', label = 'Training rmse')
    plt.plot(epoch_history, val_rmse_history, 'b', label = 'Validation rmse')
    plt.title('Training and validation rmse')
    plt.legend()
    path = os.path.join(cycle_gan.save_dir, 'accuracy.png')
    plt.savefig(path,dpi = 300)

    plt.figure()
    plt.plot(epoch_history, psnr_history, 'r', label = 'Training psnr')
    plt.plot(epoch_history, val_psnr_history, 'b', label = 'Validation psnr')
    plt.title('Training and validation psnr')
    plt.legend()
    path = os.path.join(cycle_gan.save_dir, 'psnr.png')
    plt.savefig(path, dpi = 300)

    plt.show() 
    
    # 
    np.save(os.path.join(cycle_gan.save_dir, 'epochs.npy'), np.array(epoch_history))
    np.save(os.path.join(cycle_gan.save_dir, 'losses.npy'), np.array(loss_history))
    np.save(os.path.join(cycle_gan.save_dir, 'val_losses.npy'), np.array(val_loss_history))
    np.save(os.path.join(cycle_gan.save_dir, 'rmses.npy'), np.array(rmse_history))
    np.save(os.path.join(cycle_gan.save_dir, 'val_rmses.npy'), np.array(val_rmse_history))
    np.save(os.path.join(cycle_gan.save_dir, 'psnrs.npy'), np.array(psnr_history))
    np.save(os.path.join(cycle_gan.save_dir, 'val_psnrs.npy'), np.array(val_psnr_history))

def set_seed(seed):
    random.seed(seed)
    np.random.random(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    os.environ['PYTHONHASHSEED']=str(seed)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    set_seed(cfg.seed)
    train()
