% 下采样数据集
% MS和PAN缩放因子
ratio = 4;%4;
% 卫星
sensor = 'GeoEye1';%'WV2';%'IKONOS';%'QB';%'WV3';%'none';%%%'WV3';%

% 需要处理的路径
dataset_path = 'H:\大气校正\融合\dataset\train_dataset';%'D:\实验\影像融合\data\NBU_PansharpRSData\1 Satellite_Dataset\Dataset\6 WorldView-3\train_dataset';%D:/实验/融合数据集/SpaceNet挑选/test_dataset'; 1 IKONOS 6 WorldView-3 2 QuickBird 3 GaoFen-1 4 WorldView-4 5 WorldView-2
% 原始MS存放路径
dataset_ms_path = fullfile(dataset_path, 'MS');
% 原始PAN存放路径
dataset_pan_path = fullfile(dataset_path, 'PAN');
% 下采样LR
target_ms_path = fullfile(dataset_path, 'LR');
if exist(target_ms_path)==0   %该文件夹不存在，则直接创建
    mkdir(target_ms_path);
end

% 下采样HR
target_pan_path = fullfile(dataset_path, 'HR');
if exist(target_pan_path)==0   %该文件夹不存在，则直接创建
    mkdir(target_pan_path);
end
% 下采样FR
target_gt_path = fullfile(dataset_path, 'FR');
if exist(target_gt_path)==0   %该文件夹不存在，则直接创建
    mkdir(target_gt_path);
end
% 原始MS数据列表
dataset_ms_paths = dir(fullfile(dataset_ms_path, '*.tif'));
dataset_ms_names = {dataset_ms_paths.name};
% 原始PAN数据列表
dataset_pan_paths = dir(fullfile(dataset_pan_path, '*.tif'));
dataset_pan_names = {dataset_pan_paths.name};

% 进度条
bar = waitbar(0,'准备中...');    % waitbar显示进度条
datalength = length(dataset_ms_names);
% 遍历处理
for i=1:length(dataset_ms_names)
    % 需要处理MS路径
    current_ms_path = fullfile(dataset_ms_path, dataset_ms_names(i));
    %将cell类型转换为string类型
    current_ms_path = current_ms_path{1};
    %读取MS数据
    ms_data = imread(current_ms_path);
    % 需要处理PAN路径
    current_pan_path = fullfile(dataset_pan_path, dataset_pan_names(i));
    %将cell类型转换为string类型
    current_pan_path = current_pan_path{1};
    %读取PAN数据
    pan_data = imread(current_pan_path);
    % 下采样MS和PAN
    cd ..
    [ms_down_data, pan_down_data] = resize_images(ms_data,pan_data,ratio,sensor);
    cd custum
    % 保存LR
    ms_path = fullfile(target_ms_path, dataset_ms_names(i));
    ms_path = ms_path{1};
    writeTiff(ms_down_data, ms_path);
    % 保存HR
    pan_path = fullfile(target_pan_path, dataset_pan_names(i));
    pan_path = pan_path{1};
    writeTiff(pan_down_data, pan_path);
    % 保存FR
    gt_path = fullfile(target_gt_path, dataset_ms_names(i));
    gt_path = gt_path{1};
    writeTiff(ms_data, gt_path);
    % 更新进度条
    str=['处理中...',num2str(100*i/datalength),'%'];    % 百分比形式显示处理进程,不需要删掉这行代码就行
    waitbar(i/datalength,bar,str)                      % 不注释显示
end
    