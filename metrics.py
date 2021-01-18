# -*- coding: utf-8 -*-
"""
License: GNU-3.0
Code Reference:https://github.com/wasaCheney/IQA_pansharpening_python
"""

import numpy as np
from scipy import ndimage
import cv2
import config as cfg
import math
import torch
import kornia

def get_mse(records_real, records_predict):
    """
    均方误差 估计值与真值 偏差
    """
    # if len(records_real) == len(records_predict):
    #     return sum([(x - y) ** 2 for x, y in zip(records_real, records_predict)]) / len(records_real)
    # else:
    #     return None
    records = (records_real - records_predict) ** 2
    return np.mean(records)

def get_rmse(records_real, records_predict):
    """
    均方根误差：是均方误差的算术平方根
    """
    mse = get_mse(records_real, records_predict)
    if mse:
        return math.sqrt(mse)
    else:
        return None

def get_mae(records_real, records_predict):
    """
    均方根误差：是均方误差的算术平方根
    """
    records = np.abs(records_real - records_predict)
    return np.mean(records)

def CC(X, Y): # Correlation Coefficient, th range is from -1 to +1, the ideal value is +1 
    x_mean = np.mean(X)
    y_mean = np.mean(Y)
    a = (X - x_mean) * (Y - y_mean)
    a = np.sum(a)
    b = np.sum((X - x_mean) ** 2) * np.sum((Y - y_mean) ** 2)
    b = np.sqrt(b)
    return a / (b+ np.finfo(np.float64).eps)


def sam(img1, img2):
    """SAM for 3D image, shape (H, W, C); uint or float[0, 1]"""
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    assert img1.ndim == 3 and img1.shape[2] > 1, "image n_channels should be greater than 1"
    img1_ = img1.astype(np.float64)
    img2_ = img2.astype(np.float64)
    inner_product = (img1_ * img2_).sum(axis=2)
    img1_spectral_norm = np.sqrt((img1_**2).sum(axis=2))
    img2_spectral_norm = np.sqrt((img2_**2).sum(axis=2))
    # numerical stability
    cos_theta = (inner_product / (img1_spectral_norm * img2_spectral_norm + np.finfo(np.float64).eps)).clip(min=0, max=1)
    return np.mean(np.arccos(cos_theta))


def psnr(img1, img2, dynamic_range=255):
    """PSNR metric, img uint8 if 225; uint16 if 2047"""
    if not  img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    img1_ = img1.astype(np.float64)
    img2_ = img2.astype(np.float64)
    mse = np.mean((img1_ - img2_)**2)
    if mse <= 1e-10:
        return np.inf
    return 20 * np.log10(dynamic_range / (np.sqrt(mse) + np.finfo(np.float64).eps))


# def gaussian2d(N, std):
#     t = np.arange(-(N - 1) // 2, (N + 2) // 2)
#     t1, t2 = np.meshgrid(t, t)
#     std = np.double(std)
#     w = np.exp(-0.5 * (t1 / std)**2) * np.exp(-0.5 * (t2 / std)**2) 
#     return w


# def kaiser2d(N, beta):
#     t = np.arange(-(N - 1) // 2, (N + 2) // 2) / np.double(N - 1)
#     t1, t2 = np.meshgrid(t, t)
#     t12 = np.sqrt(t1 * t1 + t2 * t2)
#     w1 = np.kaiser(N, beta)
#     w = np.interp(t12, t, w1)
#     w[t12 > t[-1]] = 0
#     w[t12 < t[0]] = 0
#     return w


# def fir_filter_wind(Hd, w):
#     """
#     compute fir (finite impulse response) filter with window method
#     Hd: desired freqeuncy response (2D)
#     w: window (2D)
#     """
#     hd = np.rot90(np.fft.fftshift(np.rot90(Hd, 2)), 2)
#     h = np.fft.fftshift(np.fft.ifft2(hd))
#     h = np.rot90(h, 2)
#     h = h * w
#     h = h / np.sum(h)
#     return h


# def GNyq2win(GNyq, scale=4, N=41):
#     """Generate a 2D convolutional window from a given GNyq
#     GNyq: Nyquist frequency
#     scale: spatial size of PAN / spatial size of MS
#     """
#     #fir filter with window method
#     fcut = 1 / scale
#     alpha = np.sqrt(((N - 1) * (fcut / 2))**2 / (-2 * np.log(GNyq)))
#     H = gaussian2d(N, alpha)
#     Hd = H / np.max(H)
#     w = kaiser2d(N, 0.5)
#     h = fir_filter_wind(Hd, w)
#     return np.real(h)


# def mtf_resize(img, satellite='QuickBird', scale=4):
#     # satellite GNyq
#     scale = int(scale)
#     if satellite == 'QuickBird':
#         GNyq = [0.34, 0.32, 0.30, 0.22]  # Band Order: B,G,R,NIR
#         GNyqPan = 0.15
#     elif satellite == 'IKONOS':
#         GNyq = [0.26, 0.28, 0.29, 0.28]  # Band Order: B,G,R,NIR
#         GNyqPan = 0.17
#     elif satellite == 'WV3':
#         GNyq = 0.29 * np.ones(8)
#         GNyqPan = 0.15
#     else:
#         raise NotImplementedError('satellite: QuickBird or IKONOS')
#     # lowpass
#     img_ = img.squeeze()
#     img_ = img_.astype(np.float64)
#     if img_.ndim == 2:  # Pan
#         H, W = img_.shape
#         lowpass = GNyq2win(GNyqPan, scale, N=41)
#     elif img_.ndim == 3:  # MS
#         H, W, _ = img.shape
#         lowpass = [GNyq2win(gnyq, scale, N=41) for gnyq in GNyq]
#         lowpass = np.stack(lowpass, axis=-1)
#     img_ = ndimage.filters.correlate(img_, lowpass, mode='nearest')
#     # downsampling
#     output_size = (H // scale, W // scale)
#     img_ = cv2.resize(img_, dsize=output_size, interpolation=cv2.INTER_NEAREST)
#     return img_


# ##################
# # No reference IQA
# ##################
# def _qindex(img1, img2, block_size=8):
#     """Q-index for 2D (one-band) image, shape (H, W); uint or float [0, 1]"""
#     assert block_size > 1, 'block_size shold be greater than 1!'
#     img1_ = img1.astype(np.float64)
#     img2_ = img2.astype(np.float64)
#     window = np.ones((block_size, block_size)) / (block_size**2)
#     # window_size = block_size**2
#     # filter, valid
#     pad_topleft = int(np.floor(block_size/2))
#     pad_bottomright = block_size - 1 - pad_topleft
#     mu1 = cv2.filter2D(img1_, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright]
#     mu2 = cv2.filter2D(img2_, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright]
#     mu1_sq = mu1**2
#     mu2_sq = mu2**2
#     mu1_mu2 = mu1 * mu2
    
#     sigma1_sq = cv2.filter2D(img1_**2, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright] - mu1_sq
#     sigma2_sq = cv2.filter2D(img2_**2, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright] - mu2_sq
# #    print(mu1_mu2.shape)
#     #print(sigma2_sq.shape)
#     sigma12 = cv2.filter2D(img1_ * img2_, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright] - mu1_mu2

#     # all = 1, include the case of simga == mu == 0
#     qindex_map = np.ones(sigma12.shape)
#     # sigma == 0 and mu != 0
    
# #    print(np.min(sigma1_sq + sigma2_sq), np.min(mu1_sq + mu2_sq))
    
#     idx = ((sigma1_sq + sigma2_sq) < 1e-8) * ((mu1_sq + mu2_sq) >1e-8)
#     qindex_map[idx] = 2 * mu1_mu2[idx] / (mu1_sq + mu2_sq)[idx]
#     # sigma !=0 and mu == 0
#     idx = ((sigma1_sq + sigma2_sq) >1e-8) * ((mu1_sq + mu2_sq) < 1e-8)
#     qindex_map[idx] = 2 * sigma12[idx] / (sigma1_sq + sigma2_sq)[idx]
#     # sigma != 0 and mu != 0
#     idx = ((sigma1_sq + sigma2_sq) >1e-8) * ((mu1_sq + mu2_sq) >1e-8)
#     qindex_map[idx] =((2 * mu1_mu2[idx]) * (2 * sigma12[idx])) / (
#         (mu1_sq + mu2_sq)[idx] * (sigma1_sq + sigma2_sq)[idx])
    
# #    print(np.mean(qindex_map))
    
# #    idx = ((sigma1_sq + sigma2_sq) == 0) * ((mu1_sq + mu2_sq) != 0)
# #    qindex_map[idx] = 2 * mu1_mu2[idx] / (mu1_sq + mu2_sq)[idx]
# #    # sigma !=0 and mu == 0
# #    idx = ((sigma1_sq + sigma2_sq) != 0) * ((mu1_sq + mu2_sq) == 0)
# #    qindex_map[idx] = 2 * sigma12[idx] / (sigma1_sq + sigma2_sq)[idx]
# #    # sigma != 0 and mu != 0
# #    idx = ((sigma1_sq + sigma2_sq) != 0) * ((mu1_sq + mu2_sq) != 0)
# #    qindex_map[idx] =((2 * mu1_mu2[idx]) * (2 * sigma12[idx])) / (
# #        (mu1_sq + mu2_sq)[idx] * (sigma1_sq + sigma2_sq)[idx])
    
#     return np.mean(qindex_map)

# def D_lambda(img_fake, img_lm, block_size=32, p=1):
#     """Spectral distortion
#     img_fake, generated HRMS
#     img_lm, LRMS"""
#     assert img_fake.ndim == img_lm.ndim == 3, 'Images must be 3D!'
#     H_f, W_f, C_f = img_fake.shape
#     H_r, W_r, C_r = img_lm.shape
#     assert C_f == C_r, 'Fake and lm should have the same number of bands!'
#     # D_lambda
#     Q_fake = []
#     Q_lm = []
#     for i in range(C_f):
#         for j in range(i+1, C_f):
#             # for fake
#             band1 = img_fake[..., i]
#             band2 = img_fake[..., j]
#             Q_fake.append(_qindex(band1, band2, block_size=block_size))
#             # for real
#             band1 = img_lm[..., i]
#             band2 = img_lm[..., j]
#             Q_lm.append(_qindex(band1, band2, block_size=block_size))
#     Q_fake = np.array(Q_fake)
#     Q_lm = np.array(Q_lm)
#     D_lambda_index = (np.abs(Q_fake - Q_lm) ** p).mean()
#     return D_lambda_index ** (1/p)


# def D_s(img_fake, img_lm, pan, satellite='QuickBird', scale=4, block_size=32, q=1):
#     """Spatial distortion
#     img_fake, generated HRMS
#     img_lm, LRMS
#     pan, HRPan"""
#     # fake and lm
#     assert img_fake.ndim == img_lm.ndim == 3, 'MS images must be 3D!'
#     H_f, W_f, C_f = img_fake.shape
#     H_r, W_r, C_r = img_lm.shape
#     assert H_f // H_r == W_f // W_r == scale, 'Spatial resolution should be compatible with scale'
#     assert C_f == C_r, 'Fake and lm should have the same number of bands!'
#     # fake and pan
#     assert pan.ndim == 3, 'Panchromatic image must be 3D!'
#     H_p, W_p, C_p = pan.shape
#     assert C_p == 1, 'size of 3rd dim of Panchromatic image must be 1'
#     assert H_f == H_p and W_f == W_p, "Pan's and fake's spatial resolution should be the same"
#     # get LRPan, 2D
#     pan_lr = mtf_resize(pan, satellite=satellite, scale=scale)
#     #print(pan_lr.shape)
#     # D_s
#     Q_hr = []
#     Q_lr = []
#     for i in range(C_f):
#         # for HR fake
#         band1 = img_fake[..., i]
#         band2 = pan[..., 0] # the input PAN is 3D with size=1 along 3rd dim
#         #print(band1.shape)
#         #print(band2.shape)
#         Q_hr.append(_qindex(band1, band2, block_size=block_size))
#         band1 = img_lm[..., i]
#         band2 = pan_lr  # this is 2D
#         #print(band1.shape)
#         #print(band2.shape)
#         Q_lr.append(_qindex(band1, band2, block_size=block_size))
#     Q_hr = np.array(Q_hr)
#     Q_lr = np.array(Q_lr)
#     D_s_index = (np.abs(Q_hr - Q_lr) ** q).mean()
#     return D_s_index ** (1/q)

# def qnr(img_fake, img_lm, pan, satellite='QuickBird', scale=4, block_size=32, p=1, q=1, alpha=1, beta=1):
#     """QNR - No reference IQA"""
#     D_lambda_idx = D_lambda(img_fake, img_lm, block_size, p)
#     D_s_idx = D_s(img_fake, img_lm, pan, satellite, scale, block_size, q)
#     QNR_idx = (1 - D_lambda_idx) ** alpha * (1 - D_s_idx) ** beta
#     return QNR_idx


# def UIQI(X, Y):
#     X_mean = torch.mean(X)
#     Y_mean = torch.mean(Y)
#     X_std = torch.std(X)
#     Y_std = torch.std(Y)
#     X_Y_cov = torch.mean((X - X_mean) * (Y - Y_mean))

#     a = X_Y_cov / (X_std * Y_std)
#     b = 2 * X_mean * Y_mean / (X_mean **2 + Y_mean**2)
#     c = 2 * X_std * Y_std / (X_std **2 + Y_std **2)
#     res = a * b * c
#     if res > 1:
#         res = 1
#     if res < -1:
#         res = -1
#     return res
def UIQI(X, Y, block_size=32): # universal image quality index
    # X_mean = torch.mean(X)
    # Y_mean = torch.mean(Y)
    # X_std = torch.std(X)
    # Y_std = torch.std(Y)
    # X_Y_cov = torch.mean((X - X_mean) * (Y - Y_mean))

    # a = X_Y_cov / (X_std * Y_std)
    # b = 2 * X_mean * Y_mean / (X_mean **2 + Y_mean**2)
    # c = 2 * X_std * Y_std / (X_std **2 + Y_std **2)
    # return a * b * c

    window = torch.ones((block_size, block_size)).cuda().unsqueeze(0) / (block_size**2)
    
    pad_topleft = int(np.floor(block_size/2))
    pad_bottomright = block_size - 1 - pad_topleft
    mu1 = kornia.filters.filter2D(X, window)[:,:,pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright]

    mu2 = kornia.filters.filter2D(Y, window)[:,:,pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = kornia.filters.filter2D(X**2, window)[:,:,pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright] - mu1_sq
    sigma2_sq = kornia.filters.filter2D(Y**2, window)[:,:,pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright] - mu2_sq
#    print(mu1_mu2.shape)
    #print(sigma2_sq.shape)
    sigma12 = kornia.filters.filter2D(X*Y, window)[:,:,pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright] - mu1_mu2

    # all = 1, include the case of simga == mu == 0
    qindex_map = torch.ones(sigma12.shape).cuda()

    
    idx = ((sigma1_sq + sigma2_sq) < 1e-8) * ((mu1_sq + mu2_sq) >1e-8)
    qindex_map[idx] = 2 * mu1_mu2[idx] / (mu1_sq + mu2_sq)[idx]
    # sigma !=0 and mu == 0
    idx = ((sigma1_sq + sigma2_sq) >1e-8) * ((mu1_sq + mu2_sq) < 1e-8)
    qindex_map[idx] = 2 * sigma12[idx] / (sigma1_sq + sigma2_sq)[idx]
    # sigma != 0 and mu != 0
    idx = ((sigma1_sq + sigma2_sq) >1e-8) * ((mu1_sq + mu2_sq) >1e-8)
    qindex_map[idx] =((2 * mu1_mu2[idx]) * (2 * sigma12[idx])) / (
        (mu1_sq + mu2_sq)[idx] * (sigma1_sq + sigma2_sq)[idx])
    

    
    return torch.mean(qindex_map)
    
def Dlambda(img_fake, img_lm, p=2):
    
    n = img_fake.shape[1]
    Q_fake = torch.zeros(int((1+n)*n/2))
    Q_lm = torch.zeros(int((1+n)*n/2))
    
    c = 0
    for i in range(n):
        for j in range(i+1, n):
            # for fake
            band1 = img_fake[:,i, :,:].unsqueeze(1)
            band2 = img_fake[:,j,:,:].unsqueeze(1)
            Q_fake[c] = UIQI(band1, band2)
            # for real
            band1 = img_lm[:,i,:,:].unsqueeze(1)
            band2 = img_lm[:,j,:,:].unsqueeze(1)
            Q_lm[c] = UIQI(band1, band2)
            c+=1
    
    D_lambda_index = (torch.abs(Q_fake - Q_lm) ** p).mean().detach().cpu()
    res = D_lambda_index ** (1/p)
    if res > 1:
        res = 1
    if res < 0:
        res = 0
    return res

def Ds(img_fake, img_lm, pan, scale, q=2):
    
    pan_lr = torch.nn.functional.interpolate(pan, scale_factor=1/scale, mode='area')

    Q_hr = torch.zeros(img_fake.shape[1])
    Q_lr = torch.zeros(img_fake.shape[1])
    for i in range(img_fake.shape[1]):
        # for HR fake
        band1 = img_fake[:,i,:,:].unsqueeze(1)
        band2 = pan # the input PAN is 3D with size=1 along 3rd dim
        
        Q_hr[i] = UIQI(band1, band2)
        band1 = img_lm[:,i,:,:].unsqueeze(1)
        band2 = pan_lr  # this is 2D
        
        Q_lr[i] = UIQI(band1, band2)
    
    D_s_index = (torch.abs(Q_hr - Q_lr) ** q).mean().detach().cpu()
    res = D_s_index ** (1/q)
    if res > 1:
        res = 1
    if res < 0:
        res = 0
    return res
        
def QNR(fuse_ms, lr_ms, hr_pan, scale=cfg.scale, alpha=1,  beta=1):

    D_lambda_idx = Dlambda(fuse_ms, lr_ms)

    
    D_s_idx = Ds(fuse_ms, lr_ms, hr_pan, scale)
    
    QNR_idx = (1 - D_lambda_idx) ** alpha * (1 - D_s_idx) ** beta
    res = QNR_idx
    if res > 1:
        res = 1
    if res < 0:
        res = 0
    return res

def gaussian2d(N, std):
    t = np.arange(-(N - 1) // 2, (N + 2) // 2)
    t1, t2 = np.meshgrid(t, t)
    std = np.double(std)
    w = np.exp(-0.5 * (t1 / std)**2) * np.exp(-0.5 * (t2 / std)**2) 
    return w


def kaiser2d(N, beta):
    t = np.arange(-(N - 1) // 2, (N + 2) // 2) / np.double(N - 1)
    t1, t2 = np.meshgrid(t, t)
    t12 = np.sqrt(t1 * t1 + t2 * t2)
    w1 = np.kaiser(N, beta)
    w = np.interp(t12, t, w1)
    w[t12 > t[-1]] = 0
    w[t12 < t[0]] = 0
    return w


def fir_filter_wind(Hd, w):
    """
    compute fir (finite impulse response) filter with window method
    Hd: desired freqeuncy response (2D)
    w: window (2D)
    """
    hd = np.rot90(np.fft.fftshift(np.rot90(Hd, 2)), 2)
    h = np.fft.fftshift(np.fft.ifft2(hd))
    h = np.rot90(h, 2)
    h = h * w
    h = h / np.sum(h)
    return h


def GNyq2win(GNyq, scale=4, N=41):
    """Generate a 2D convolutional window from a given GNyq
    GNyq: Nyquist frequency
    scale: spatial size of PAN / spatial size of MS
    """
    #fir filter with window method
    fcut = 1 / scale
    alpha = np.sqrt(((N - 1) * (fcut / 2))**2 / (-2 * np.log(GNyq)))
    H = gaussian2d(N, alpha)
    Hd = H / np.max(H)
    w = kaiser2d(N, 0.5)
    h = fir_filter_wind(Hd, w)
    return np.real(h)


def mtf_resize(img, satellite='QuickBird', scale=4):
    # satellite GNyq
    scale = int(scale)
    if satellite == 'QuickBird':
        GNyq = [0.34, 0.32, 0.30, 0.22]  # Band Order: B,G,R,NIR
        GNyqPan = 0.15
    elif satellite == 'IKONOS':
        GNyq = [0.26, 0.28, 0.29, 0.28]  # Band Order: B,G,R,NIR
        GNyqPan = 0.17
    elif satellite == 'WV3':
        GNyq = 0.29 * np.ones(8)
        GNyqPan = 0.15
    else:
        raise NotImplementedError('satellite: QuickBird or IKONOS')
    # lowpass
    img_ = img.squeeze()
    img_ = img_.astype(np.float64)
    if img_.ndim == 2:  # Pan
        H, W = img_.shape
        lowpass = GNyq2win(GNyqPan, scale, N=41)
    elif img_.ndim == 3:  # MS
        H, W, _ = img.shape
        lowpass = [GNyq2win(gnyq, scale, N=41) for gnyq in GNyq]
        lowpass = np.stack(lowpass, axis=-1)
    img_ = ndimage.filters.correlate(img_, lowpass, mode='nearest')
    # downsampling
    output_size = (H // scale, W // scale)
    img_ = cv2.resize(img_, dsize=output_size, interpolation=cv2.INTER_NEAREST)
    return img_


##################
# No reference IQA
##################

def _qindex(img1, img2, block_size=8):
    """Q-index for 2D (one-band) image, shape (H, W); uint or float [0, 1]"""
    assert block_size > 1, 'block_size shold be greater than 1!'
    img1_ = img1.astype(np.float64)
    img2_ = img2.astype(np.float64)
    window = np.ones((block_size, block_size)) / (block_size**2)
    # window_size = block_size**2
    # filter, valid
    pad_topleft = int(np.floor(block_size/2))
    pad_bottomright = block_size - 1 - pad_topleft
    mu1 = cv2.filter2D(img1_, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright]
    mu2 = cv2.filter2D(img2_, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.filter2D(img1_**2, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright] - mu1_sq
    sigma2_sq = cv2.filter2D(img2_**2, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright] - mu2_sq
#    print(mu1_mu2.shape)
    #print(sigma2_sq.shape)
    sigma12 = cv2.filter2D(img1_ * img2_, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright] - mu1_mu2

    # all = 1, include the case of simga == mu == 0
    qindex_map = np.ones(sigma12.shape)
    # sigma == 0 and mu != 0
    
#    print(np.min(sigma1_sq + sigma2_sq), np.min(mu1_sq + mu2_sq))
    
    idx = ((sigma1_sq + sigma2_sq) < 1e-8) * ((mu1_sq + mu2_sq) >1e-8)
    qindex_map[idx] = 2 * mu1_mu2[idx] / (mu1_sq + mu2_sq)[idx]
    # sigma !=0 and mu == 0
    idx = ((sigma1_sq + sigma2_sq) >1e-8) * ((mu1_sq + mu2_sq) < 1e-8)
    qindex_map[idx] = 2 * sigma12[idx] / (sigma1_sq + sigma2_sq)[idx]
    # sigma != 0 and mu != 0
    idx = ((sigma1_sq + sigma2_sq) >1e-8) * ((mu1_sq + mu2_sq) >1e-8)
    qindex_map[idx] =((2 * mu1_mu2[idx]) * (2 * sigma12[idx])) / (
        (mu1_sq + mu2_sq)[idx] * (sigma1_sq + sigma2_sq)[idx])
    
#    print(np.mean(qindex_map))
    
#    idx = ((sigma1_sq + sigma2_sq) == 0) * ((mu1_sq + mu2_sq) != 0)
#    qindex_map[idx] = 2 * mu1_mu2[idx] / (mu1_sq + mu2_sq)[idx]
#    # sigma !=0 and mu == 0
#    idx = ((sigma1_sq + sigma2_sq) != 0) * ((mu1_sq + mu2_sq) == 0)
#    qindex_map[idx] = 2 * sigma12[idx] / (sigma1_sq + sigma2_sq)[idx]
#    # sigma != 0 and mu != 0
#    idx = ((sigma1_sq + sigma2_sq) != 0) * ((mu1_sq + mu2_sq) != 0)
#    qindex_map[idx] =((2 * mu1_mu2[idx]) * (2 * sigma12[idx])) / (
#        (mu1_sq + mu2_sq)[idx] * (sigma1_sq + sigma2_sq)[idx])
    
    return np.mean(qindex_map)

def D_lambda(img_fake, img_lm, block_size=32, p=1):
    """Spectral distortion
    img_fake, generated HRMS
    img_lm, LRMS"""
    assert img_fake.ndim == img_lm.ndim == 3, 'Images must be 3D!'
    H_f, W_f, C_f = img_fake.shape
    H_r, W_r, C_r = img_lm.shape
    assert C_f == C_r, 'Fake and lm should have the same number of bands!'
    # D_lambda
    Q_fake = []
    Q_lm = []
    for i in range(C_f):
        for j in range(i+1, C_f):
            # for fake
            band1 = img_fake[..., i]
            band2 = img_fake[..., j]
            Q_fake.append(_qindex(band1, band2, block_size=block_size))
            # for real
            band1 = img_lm[..., i]
            band2 = img_lm[..., j]
            Q_lm.append(_qindex(band1, band2, block_size=block_size))
    Q_fake = np.array(Q_fake)
    Q_lm = np.array(Q_lm)
    D_lambda_index = (np.abs(Q_fake - Q_lm) ** p).mean()
    return D_lambda_index ** (1/p)


def D_s(img_fake, img_lm, pan, satellite='QuickBird', scale=4, block_size=32, q=1):
    """Spatial distortion
    img_fake, generated HRMS
    img_lm, LRMS
    pan, HRPan"""
    # fake and lm
    assert img_fake.ndim == img_lm.ndim == 3, 'MS images must be 3D!'
    H_f, W_f, C_f = img_fake.shape
    H_r, W_r, C_r = img_lm.shape
    assert H_f // H_r == W_f // W_r == scale, 'Spatial resolution should be compatible with scale'
    assert C_f == C_r, 'Fake and lm should have the same number of bands!'
    # fake and pan
    assert pan.ndim == 3, 'Panchromatic image must be 3D!'
    H_p, W_p, C_p = pan.shape
    assert C_p == 1, 'size of 3rd dim of Panchromatic image must be 1'
    assert H_f == H_p and W_f == W_p, "Pan's and fake's spatial resolution should be the same"
    # get LRPan, 2D
    pan_lr = mtf_resize(pan, satellite=satellite, scale=scale)
    #print(pan_lr.shape)
    # D_s
    Q_hr = []
    Q_lr = []
    for i in range(C_f):
        # for HR fake
        band1 = img_fake[..., i]
        band2 = pan[..., 0] # the input PAN is 3D with size=1 along 3rd dim
        #print(band1.shape)
        #print(band2.shape)
        Q_hr.append(_qindex(band1, band2, block_size=block_size))
        band1 = img_lm[..., i]
        band2 = pan_lr  # this is 2D
        #print(band1.shape)
        #print(band2.shape)
        Q_lr.append(_qindex(band1, band2, block_size=block_size))
    Q_hr = np.array(Q_hr)
    Q_lr = np.array(Q_lr)
    D_s_index = (np.abs(Q_hr - Q_lr) ** q).mean()
    return D_s_index ** (1/q)

def qnr(img_fake, img_lm, pan, satellite='QuickBird', scale=4, block_size=32, p=1, q=1, alpha=1, beta=1):
    """QNR - No reference IQA"""
    D_lambda_idx = D_lambda(img_fake, img_lm, block_size, p)
    D_s_idx = D_s(img_fake, img_lm, pan, satellite, scale, block_size, q)
    QNR_idx = (1 - D_lambda_idx) ** alpha * (1 - D_s_idx) ** beta
    return QNR_idx