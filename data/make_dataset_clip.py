import gdal
import os
import glob
import numpy as np
import random
from tqdm import tqdm
import shutil



def read_tiff(input_file):
    """
    读取影像
    :param input_file:输入影像
    :return:波段数据，仿射变换参数，投影信息、行数、列数、波段数
    """

    dataset = gdal.Open(input_file)
    rows = dataset.RasterYSize
    cols = dataset.RasterXSize

    geo = dataset.GetGeoTransform()
    proj = dataset.GetProjection()

    couts = dataset.RasterCount
    if (satellite == 'wv3' or satellite == 'wv2') and couts == 8:
        bands = [2,3,5,7] # b, g, r, nir
        array_data = np.zeros((4,rows,cols))

        for i in range(len(bands)):
            band = dataset.GetRasterBand(bands[i])
            array_data[i,:,:] = band.ReadAsArray()
        couts = 4
    else:
        array_data = dataset.ReadAsArray()

    return array_data,geo,proj,rows,cols,couts
    
def write_tiff(output_file,array_data,rows,cols,counts,geo,proj):

    #判断栅格数据的数据类型
    
    if 'int8' in array_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in array_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    Driver = gdal.GetDriverByName("Gtiff")
    dataset = Driver.Create(output_file,cols,rows,counts,datatype)

    dataset.SetGeoTransform(geo)
    dataset.SetProjection(proj)

    if len(array_data.shape) == 2:
        array_data = array_data.reshape(1, array_data.shape[0], array_data.shape[1])
    for i in range(counts):
        band = dataset.GetRasterBand(i+1)
        band.WriteArray(array_data[i,:,:])
  
  

def copy(source, target):
    array_data,geo,proj,rows,cols,couts = read_tiff(source)
    write_tiff(target,array_data,rows,cols,couts,geo,proj)

def pixel2world(geo, x, y):
    Xgeo = geo[0] + x*geo[1]+ y*geo[2]
    Ygeo = geo[3] + x*geo[4] + y*geo[5]
    return Xgeo, Ygeo

def world2Pixel(geoMatrix, x, y):
    """
    Uses a gdal geomatrix (gdal.GetGeoTransform()) to calculate
    the pixel location of a geospatial coordinate
    """
    # ulX = geoMatrix[0]
    # ulY = geoMatrix[3]
    # xDist = geoMatrix[1]
    # pixel = int((x - ulX) / xDist)
    # line = int((ulY - y) / xDist)
    # print(pixel, line)
    # return (pixel, line)
    dTemp = geoMatrix[1] * geoMatrix[5] - geoMatrix[2] *geoMatrix[4]
    Xpixel= (geoMatrix[5] * (x - geoMatrix[0]) -geoMatrix[2] * (y - geoMatrix[3])) / dTemp + 0.5
    Yline = (geoMatrix[1] * (y - geoMatrix[3]) -geoMatrix[4] * (x - geoMatrix[0])) / dTemp + 0.5
    # print(Xpixel, Yline)
    return (int(Xpixel), int(Yline))



# would like to update to clip image by the format of 16 bit
satellite = '' #'wv2'
input_data_path = r'H:\shouxian_sentinel\T50SMA\s2_pansharpen_data'
input_ms_path = glob.glob(os.path.join(input_data_path, "mul", "*.TIF"))#r'./ms.tif'
input_pan_path = glob.glob(os.path.join(input_data_path, "pan", "*.TIF"))#r'./pan.tif'
print(input_ms_path, input_pan_path)
# print('共有{}对影像'.format(len(input_pan_path)))
# 2304 is for scale 6
clip_size = 2304#1024#256#128#50 #112
scale = 6
isRandom = False
align_by_geo = True
    
            


# 开始对训练集切割

print('裁剪')
random.seed(19970716)
save_path = os.path.join(input_data_path, 'dataset')
if not os.path.exists(save_path):
    os.makedirs(save_path)

save_hr_path = os.path.join(save_path, 'PAN')
save_lr_path = os.path.join(save_path, 'MS')

if not os.path.exists(save_hr_path):
    os.makedirs(save_hr_path)
if not os.path.exists(save_lr_path):
    os.makedirs(save_lr_path)


# make dataset
data_index = 0 # global data index
for i, ms_path in enumerate(input_ms_path):
    print(i)
    lr_array_data,lr_geo,lr_proj,lr_rows,lr_cols,lr_couts = read_tiff(ms_path)
    array_data,geo,proj,rows,cols,couts = read_tiff(input_pan_path[i])

    # resize_array_data = GDAL_RESIZE(lr_array_data, cols, rows, interpolation=cv2.INTER_LINEAR)
    j_iters = lr_rows // (clip_size // scale)
    k_iters = lr_cols // (clip_size // scale)

    print(array_data.shape)
    print(lr_array_data.shape)

    for j in range(j_iters):
        for k in range(k_iters):
            size = clip_size // scale
            h_s = j * size
            h_e = h_s + size
            w_s = k * size
            w_e = w_s + size
            ms_data = lr_array_data[:,h_s:h_e, w_s:w_e].astype(np.uint16)
            
            if align_by_geo:
                x, y = pixel2world(lr_geo, w_s, h_s) # this function should tranfer the format of (w, h) not (h,w)
                lr_b_geo = list(lr_geo)
                lr_b_geo[0] = x
                lr_b_geo[3] = y

            
                (m_y,m_x) = world2Pixel(geo, x, y) # x is lon, y is lat, but out is (col, row)
                b_geo = list(geo)
                b_geo[0] = x
                b_geo[3] = y

                
                pan_data = array_data[:, (m_x):(m_x)+clip_size, (m_y):(m_y)+clip_size]
                
            else:
                pan_data = array_data[:, h_s*scale:h_e*scale, w_s*scale:w_e*scale]
                lr_b_geo = list(lr_geo)
                b_geo = list(geo)

            if np.sum(ms_data!=0) >= int(ms_data.shape[0] * ms_data.shape[1] * ms_data.shape[2]):
                # 路径
                save_lr_batch_path = os.path.join(save_lr_path, '{}.tif'.format(data_index))
                save_hr_batch_path = os.path.join(save_hr_path, '{}.tif'.format(data_index))
            

                # 保存
                write_tiff(save_lr_batch_path, ms_data, size, size, lr_couts, tuple(lr_b_geo), lr_proj)  
                write_tiff(save_hr_batch_path, pan_data, clip_size, clip_size, couts, tuple(b_geo), proj)
            
                data_index +=1
    

   
