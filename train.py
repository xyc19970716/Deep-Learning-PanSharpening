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
from thop import profile
from ptflops import get_model_complexity_info
import argparse


def get_dataset(args):

    data_train = PsDataset(args, apath=args.dataDir, isUnlabel=args.isUnlabel)#PsRamDataset(apath=cfg.dataDir, isUnlabel=cfg.isUnlabel)#LmdbDataset()#
    if args.isEval:
        train_data, test_data = Get_DataSet(data_train, [0.8, 0.2])
        dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batchSize,
                                                drop_last=True, shuffle=True, num_workers=int(args.nThreads),
                                                pin_memory=True)
        dataloader2 = torch.utils.data.DataLoader(test_data, batch_size=args.batchSize,
                                                drop_last=True, shuffle=False, num_workers=int(args.nThreads),
                                                pin_memory=True)
        return dataloader, dataloader2
    else:
        dataloader = torch.utils.data.DataLoader(data_train, batch_size=args.batchSize,
                                                drop_last=True, shuffle=True, num_workers=int(args.nThreads),
                                                pin_memory=True)
        return dataloader

def evalOrSaveBest(net, dataloader, best_eval_index, args):
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
            
            fake_B = net.fake_B.cpu().detach().numpy() * args.data_range
            real_B = net.real_B.cpu().detach().numpy() * args.data_range
            
            current_batch_eval_index = metrics.get_rmse(real_B, fake_B) 

            current_eval_index += current_batch_eval_index
            
            print('Valing: {}'.format(step), 'current_rmse: {}'.format(current_batch_eval_index / args.batchSize))
            step += 1
            
            
            
            val_loss += net.loss_G
            val_rmse += current_batch_eval_index
            val_psnr += metrics.psnr(fake_B, real_B, dynamic_range=args.data_range)
        

    # print(len(dataloader))
    current_eval_index = current_eval_index / len(dataloader) / args.batchSize
    
    print('val_rmse=', current_eval_index)
    if current_eval_index < best_eval_index:
        print('better than best_rmse=', best_eval_index, 'save to best')
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

def train(args):
    log = LossLog(args)

    #  select network
    if args.model == 'ours':
        cycle_gan = OursModel()
    elif args.model == 'PNN':
        cycle_gan = PNNModel()
    elif args.model == 'PanNet':
        cycle_gan = PanNetModel()
    elif args.model == 'TFNet':
        cycle_gan = TFNetModel()
    elif args.model == 'PSGAN':
        cycle_gan = PSGANModel()
    elif args.model == 'MSDCNN':
        cycle_gan = MSDCNNModel()
    elif args.model == 'SRPPNN':
        cycle_gan = SRPPNNModel()
  

    print(cycle_gan)
    cycle_gan.initialize(args)
    # cycle_gan.cuda()

    cycle_gan.setup()
    input1 = torch.randn(1, args.mul_channel, 64, 64).cuda()
    input2 = torch.randn(1, args.pan_channel, 256, 256).cuda()
    flop, para = profile(cycle_gan.netG, inputs=(input1, input2, ))
    print("%.2fM" % (flop/1e6), "%.2fM" % (para/1e6))
  
    print('Total params: %.2fM' % (sum(p.numel() for p in cycle_gan.netG.parameters())/1000000.0))

    # load data
    if args.isEval:
        dataloader, dataloader2 = get_dataset(args)
    else:
        dataloader = get_dataset(args)

    batch_iter = 0 # iterations
    lr_decay_iters_idx = 0


    cycle_gan.train()
    
    best_psnr = 999999

    # mse
    mse = nn.MSELoss()
    # 计算一代的迭代次数
    epoch_iter_nums = len(dataloader)
    # 计算总共迭代次数
    total_iter_nums = epoch_iter_nums * args.epochs

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
    epoch_history = [i+1 for i in range(args.epochs)]


    for epoch in range(args.which_epoch+1, args.epochs):
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
            avg_psnr += metrics.psnr(cycle_gan.fake_B.detach().cpu().numpy()*args.data_range, cycle_gan.real_B.detach().cpu().numpy()*args.data_range, dynamic_range=args.data_range)
            # 打印信息
            if (batch_iter+1) % args.print_freq == 0: 
                t = (time.time() - iter_start_time) / args.batchSize
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
        change_infos = cycle_gan.update_learning_rate(decay_factor=args.lr_decay_factor)
        # 显示并记录日志
        for info in change_infos:
            log.print_change_learning_rate(epoch, info['name'], info['old_lr'], info['new_lr'])

        if args.isEval:
            best_psnr, avg_val_loss, avg_val_rmse, avg_val_psnr = evalOrSaveBest(net=cycle_gan, dataloader=dataloader2, best_eval_index=best_psnr, args=args)  
        else:
            best_psnr, avg_val_loss, avg_val_rmse, avg_val_psnr = 0,0,0,0
        val_loss_history.append(avg_val_loss)
        val_rmse_history.append(avg_val_rmse)
        val_psnr_history.append(avg_val_psnr)

        if (epoch+1) % args.save_epoch_freq == 0:
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

    # plt.show() 
    
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
    parser.add_argument("-sf", "--save_epoch_freq", type=int, default=100, help="print frequency of log")
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
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    set_seed(args.seed)
    train(args)
