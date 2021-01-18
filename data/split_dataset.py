import gdal
import os
import glob
import numpy as np
import random



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
input_data_path = r'./'
input_ms_path = glob.glob(os.path.join(os.path.join(input_data_path, 'MS'), '*.tif'))
input_pan_path = glob.glob(os.path.join(os.path.join(input_data_path, 'PAN'), '*.tif'))
print(input_ms_path, input_pan_path)
print('共有{}对影像'.format(len(input_pan_path)))


            
# 先开始随机打乱数据集
test_path = r'./test_dataset'
if not os.path.exists(test_path):
    os.makedirs(test_path)

test_hr_path = os.path.join(test_path, 'PAN')
test_lr_path = os.path.join(test_path, 'MS')

if not os.path.exists(test_hr_path):
    os.makedirs(test_hr_path)
if not os.path.exists(test_lr_path):
    os.makedirs(test_lr_path)

train_temp_path = r'./train_dataset'
if not os.path.exists(train_temp_path):
    os.makedirs(train_temp_path)

train_temp_hr_path = os.path.join(train_temp_path, 'PAN')
train_temp_lr_path = os.path.join(train_temp_path, 'MS')

if not os.path.exists(train_temp_hr_path):
    os.makedirs(train_temp_hr_path)
if not os.path.exists(train_temp_lr_path):
    os.makedirs(train_temp_lr_path)

spilt_factor = 0.3




random.seed(0)
random.shuffle(input_ms_path)
random.seed(0)
random.shuffle(input_pan_path)


border_idx = int(len(input_ms_path)-200)#int(len(input_ms_path) * (1-spilt_factor))#

# train
print('开始随机分割数据集')
print("train set")
for i in tqdm(range(0, border_idx)):
    ms_name = os.path.basename(input_ms_path[i])
    pan_name = os.path.basename(input_pan_path[i])
  
    ms_path = os.path.join(train_temp_lr_path, ms_name)
    pan_path = os.path.join(train_temp_hr_path, pan_name)

    # shutil.copy(input_ms_path[i], ms_path)
    # shutil.copy(input_pan_path[i], pan_path)
    copy(input_ms_path[i], ms_path)
    copy(input_pan_path[i], pan_path)

# test
print("test set")
for i in tqdm(range(border_idx, len(input_ms_path))):
    ms_name = os.path.basename(input_ms_path[i])
    pan_name = os.path.basename(input_pan_path[i])
  
    ms_path = os.path.join(test_lr_path, ms_name)
    pan_path = os.path.join(test_hr_path, pan_name)

    # shutil.copy(input_ms_path[i], ms_path)
    # shutil.copy(input_pan_path[i], pan_path)
    copy(input_ms_path[i], ms_path)
    copy(input_pan_path[i], pan_path)



   
