import os
import os.path
import torch
import numpy as np
import cv2
from scipy import ndimage
from scipy import signal
import scipy.misc as misc
import datetime
import re
import matplotlib.pyplot as plt

def findvalue(line, name):
    value = re.findall(r'{}: -?\d+\.?\d*E?-?\d+\d+?|{}: \d+'.format(name, name), line)
    value = re.findall(r'-?\d+\.?\d*E?-?\d+\d+?|\d+', value[0])
    return value[0]

class LossLog():
    def __init__(self, args):
        # 初始化log,定义存储位置
        self.save_dir = os.path.join(args.checkpoints_dir, args.model+args.model_sub+args.model_loss)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if args.isUnlabel:
            self.save_dir = os.path.join(self.save_dir, 'unsupervised')
        else:
            self.save_dir = os.path.join(self.save_dir, 'supervised')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        if os.path.exists(self.save_dir + '/log.txt'):
            self.logFile = open(self.save_dir + '/log.txt', 'a')
        else:
            self.logFile = open(self.save_dir + '/log.txt', 'w')
        
        # self.logFile = open(self.save_dir + '/log.txt', 'w')

    def print_current_losses(self, epoch, batch, epoch_iter_num, i, total_iter_num, losses, t, t_data):
        # 显示当前losses，写入日志
        time = datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S  %A')
        message = '[INFO] %s ' % (str(time))
        message += '(epoch: %d, batch: %d/%d, iters: %d/%d, time: %.3f, data: %.3f) ' % (epoch, batch, epoch_iter_num, i, total_iter_num, t, t_data)
        for k, v in losses.items():
            message += '%s: %.6f ' % (k, v)

        print(message)
        self.logFile.write(message + '\n')

    def print_change_learning_rate(self, epoch, name, old_lr, lr):
        # 显示改变学习率操作，写入日志
        time = datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S  %A')
        message = '[INFO] %s ' % (str(time))
        message += '(epoch: %d, optimizer_%s, learning_rate: %.7f to %.7f) ' % (epoch, name, old_lr, lr)
        print(message)
        self.logFile.write(message + '\n')




if __name__ == "__main__":
    log = LossLog()
   