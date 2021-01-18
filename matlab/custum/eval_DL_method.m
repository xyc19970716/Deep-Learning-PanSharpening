% 评价全和降的
% 参数
ratio = 4;
Qblocks_size = 32;
L = 11;
flag_cut_bounds = 1;
dim_cut = 11;
thvalues = 0;
sensor = 'IKONOS';%'QB';%'WV2';%'GeoEye1';%'IKONOS';%''WV3';%'none';%'WV3';%'none';% %'none';%'IKONOS';%'WV3';%'none';%'WV3';'none';% %设置
im_tag = 'none';%8;%'Tls1';%'none'; % 设置
% 需要处理的路径
test_dataset_path = 'D:\实验\影像融合\IKONOS\test_dataset';%'D:\实验\影像融合\data\NBU_PansharpRSData\1 Satellite_Dataset\Dataset\6 WorldView-3\test_dataset';%'D:/实验/融合数据集/SpaceNet挑选/test_dataset';
test_dataset_path2 = 'D:\实验\影像融合\IKONOS\result\CycleMP\supervised';%'D:\实验\影像融合\data\NBU_PansharpRSData\1 Satellite_Dataset\Dataset\6 WorldView-3\result\Ours\supervised';%'D:/实验/PsCycleGAN/result/PanNet';5 WorldView-2 2 QuickBird 4 WorldView-4 6 WorldView-3 3 GaoFen-1 1 IKONOS
save_path = fullfile(test_dataset_path, 'OursDIP.mat');
% 下采样LR路径
test_ms_path = fullfile(test_dataset_path, 'LR');
% 下采样HR
test_pan_path = fullfile(test_dataset_path, 'HR');
% 下采样FR
test_gt_path = fullfile(test_dataset_path, 'FR');
% 预测
test_fusion_path = fullfile(test_dataset_path2, 'reduce'); % 设置
% 原始MS
test_origin_ms_path = fullfile(test_dataset_path, 'MS');
% 原始PAN
test_origin_pan_path = fullfile(test_dataset_path, 'PAN');
% 原始融合
test_origin_fusion_path = fullfile(test_dataset_path2, 'full'); %设置

% 下采样MS数据列表
test_ms_paths = dir(fullfile(test_ms_path, '*.tif'));
test_ms_names = {test_ms_paths.name};
% 下采样PAN数据列表
test_pan_paths = dir(fullfile(test_pan_path, '*.tif'));
test_pan_names = {test_pan_paths.name};
% GT数据列表
test_gt_paths = dir(fullfile(test_gt_path, '*.tif'));
test_gt_names = {test_gt_paths.name};
% 融合数据列表
test_fusion_paths = dir(fullfile(test_fusion_path, '*.tif'));
test_fusion_names = {test_fusion_paths.name};
% 原始MS数据列表
test_origin_ms_paths = dir(fullfile(test_origin_ms_path, '*.tif'));
test_origin_ms_names = {test_origin_ms_paths.name};
% 原始PAN数据列表
test_origin_pan_paths = dir(fullfile(test_origin_pan_path, '*.tif'));
test_origin_pan_names = {test_origin_pan_paths.name};
% 原始Fusion数据列表
test_origin_fusion_paths = dir(fullfile(test_origin_fusion_path, '*.tif'));
test_origin_fusion_names = {test_origin_fusion_paths.name};


% 进度条
bar = waitbar(0,'准备中...');    % waitbar显示进度条
datalength = length(test_fusion_names);
% 初始化评价指标数组
Q_avgS = zeros(1, datalength);
SAMS = zeros(1, datalength);
ERGASS = zeros(1, datalength);
SCC_GTS = zeros(1, datalength);
QS = zeros(1, datalength);
dll = zeros(1, datalength);
dss = zeros(1, datalength);
qnrs = zeros(1, datalength);
% 遍历处理
for i=1:datalength
    % 融合
    current_test_fusion_path = fullfile(test_fusion_path, test_fusion_names(i));
    %将cell类型转换为string类型
    current_test_fusion_path = current_test_fusion_path{1};
    %读取融合数据
    fusion_data = imread(current_test_fusion_path);
    
    % GT
    current_test_gt_path = fullfile(test_gt_path, test_gt_names(i));
    %将cell类型转换为string类型
    current_test_gt_path = current_test_gt_path{1};
    %读取融合数据
    gt_data = imread(current_test_gt_path);
    
    % MS
    current_test_ms_path = fullfile(test_ms_path, test_ms_names(i));
    %将cell类型转换为string类型
    current_test_ms_path = current_test_ms_path{1};
    %读取融合数据
    ms_data = imread(current_test_ms_path);
    
    % PAN
    current_test_pan_path = fullfile(test_pan_path, test_pan_names(i));
    %将cell类型转换为string类型
    current_test_pan_path = current_test_pan_path{1};
    %读取融合数据
    pan_data = imread(current_test_pan_path);
    
    % 计算评价
    cd ..
    [Q_avg, SAM, ERGAS, SCC_GT, Q] = indexes_evaluation(double(fusion_data),double(gt_data),ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
    Q_avgS(i) = Q_avg;
    SAMS(i) = SAM;
    ERGASS(i) = ERGAS;
    SCC_GTS(i) = SCC_GT;
    QS(i) = Q;
    cd custum
    
    I_MS_UP = zeros(round(size(ms_data,1)*ratio),round(size(ms_data,2)*ratio),size(ms_data,3));
    for idim=1:size(ms_data,3)
        I_MS_UP(:,:,idim) = imresize(ms_data(:,:,idim),ratio);
    end
    

    % 计算评价
    cd ..
    [D_lambda,D_S,QNR_index,SAM_index,sCC] = indexes_evaluation_FS(double(fusion_data),double(ms_data),double(pan_data),L,thvalues,double(I_MS_UP),sensor,im_tag,ratio);
    dll(i) = D_lambda;
    dss(i) = D_S;
    qnrs(i) = QNR_index;
    cd custum
    
    name = test_fusion_names(i);
    name = name{1};
    fprintf('%s->Q_avg:%0.4f, SAM:%0.4f, ERGAS:%0.4f, SCC_GT:%0.4f, Q:%0.4f\n',name, Q_avg, SAM, ERGAS, SCC_GT, Q);
    
    % 更新进度条
    str=['处理中...',num2str(100*i/datalength),'%'];    % 百分比形式显示处理进程,不需要删掉这行代码就行
    waitbar(i/datalength,bar,str)                      % 不注释显示
end
% 计算各个指标的均值和方差
Q_avg_mean = nanmean(Q_avgS);
SAM_mean = nanmean(SAMS);
ERGAS_mean = nanmean(ERGASS);
SCC_GT_mean = nanmean(SCC_GTS);
Q_mean = nanmean(QS);

Q_avg_var = nanstd(Q_avgS);
SAM_var = nanstd(SAMS);
ERGAS_var = nanstd(ERGASS);
SCC_GT_var = nanstd(SCC_GTS);
Q_var = nanstd(QS);

dll_mean = nanmean(dll);
dss_mean = nanmean(dss);
qnrs_mean = nanmean(qnrs);

dll_var = nanstd(dll);
dss_var = nanstd(dss);
qnrs_var = nanstd(qnrs);

% 开始全分辨率评价
% 进度条
bar = waitbar(0,'准备中...');    % waitbar显示进度条
datalength = length(test_origin_fusion_names);
% 初始化评价指标数组
D_lambdaS = zeros(1, datalength);
D_SS = zeros(1, datalength);
QNR_indexS = zeros(1, datalength);
SAM_indexS = zeros(1, datalength);
sCCS = zeros(1, datalength);

Q_avgFS = zeros(1, datalength);
SAMFS = zeros(1, datalength);
ERGASFS = zeros(1, datalength);
SCC_GTFS = zeros(1, datalength);
QFS = zeros(1, datalength);

% 遍历处理
for i=1:datalength
    % 融合
    current_test_origin_fusion_path = fullfile(test_origin_fusion_path, test_origin_fusion_names(i));
    %将cell类型转换为string类型
    current_test_origin_fusion_path = current_test_origin_fusion_path{1};
    %读取融合数据
    fusion_data = imread(current_test_origin_fusion_path);
 
    % MS
    current_test_origin_ms_path = fullfile(test_origin_ms_path, test_origin_ms_names(i));
    %将cell类型转换为string类型
    current_test_origin_ms_path = current_test_origin_ms_path{1};
    %读取融合数据
    ms_data = imread(current_test_origin_ms_path);

    % PAN
    current_test_origin_pan_path = fullfile(test_origin_pan_path, test_origin_pan_names(i));
    %将cell类型转换为string类型
    current_test_origin_pan_path = current_test_origin_pan_path{1};
    %读取融合数据
    pan_data = imread(current_test_origin_pan_path);

    % 上采样
    %%% Bicubic Interpolator MS
    %cd ..
    %I_MS_UP = interp23tap(ms_data, ratio);
    %cd custum
    I_MS_UP = zeros(round(size(ms_data,1)*ratio),round(size(ms_data,2)*ratio),size(ms_data,3));
    for idim=1:size(ms_data,3)
        I_MS_UP(:,:,idim) = imresize(ms_data(:,:,idim),ratio);
    end
    

    % 计算评价
    cd ..
    [D_lambda,D_S,QNR_index,SAM_index,sCC] = indexes_evaluation_FS(double(fusion_data),double(ms_data),double(pan_data),L,thvalues,double(I_MS_UP),sensor,im_tag,ratio);
    D_lambdaS(i) = D_lambda;
    D_SS(i) = D_S;
    QNR_indexS(i) = QNR_index;
    SAM_indexS(i) = SAM_index;
    sCCS(i) = sCC;
    cd custum
    name = test_origin_fusion_names(i);
    name = name{1};
    fprintf('%s->D_l:%0.4f,D_s:%0.4f,QNR:%0.4f\n',name, D_lambda, D_S,QNR_index);
    % 下采样
    %%% Bicubic Interpolator MS
    %fusion_down_data = zeros(round(size(fusion_data,1)/ratio),round(size(fusion_data,2)/ratio),size(fusion_data,3));
    %for idim=1:size(fusion_data,3)
    %    fusion_down_data(:,:,idim) = imresize(fusion_data(:,:,idim),1/ratio);
    %end
    cd ..
    [fusion_down_data, pan_down_data] = resize_images(fusion_data,pan_data,ratio,sensor);
    cd custum
    % 计算评价
    cd ..
    [Q_avg, SAM, ERGAS, SCC_GT, Q] = indexes_evaluation(double(fusion_down_data),double(ms_data),ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
    Q_avgFS(i) = Q_avg;
    SAMFS(i) = SAM;
    ERGASFS(i) = ERGAS;
    SCC_GTFS(i) = SCC_GT;
    QFS(i) = Q;
    cd custum
    
    % 更新进度条
    str=['处理中...',num2str(100*i/datalength),'%'];    % 百分比形式显示处理进程,不需要删掉这行代码就行
    waitbar(i/datalength,bar,str)                      % 不注释显示
end
% 初始化评价指标数组
D_lambda_mean = nanmean(D_lambdaS);
D_S_mean = nanmean(D_SS);
QNR_index_mean = nanmean(QNR_indexS);
SAM_index_mean = nanmean(SAM_indexS);
sCC_mean = nanmean(sCCS);

D_lambda_var = nanstd(D_lambdaS);
D_S_var = nanstd(D_SS);
QNR_index_var = nanstd(QNR_indexS);
SAM_index_var = nanstd(SAM_indexS);
sCC_var = nanstd(sCCS);

% 计算各个指标的均值和方差
Q_avgF_mean = nanmean(Q_avgFS);
SAMF_mean = nanmean(SAMFS);
ERGASF_mean = nanmean(ERGASFS);
SCC_GTF_mean = nanmean(SCC_GTFS);
QF_mean = nanmean(QFS);

Q_avgF_var = nanstd(Q_avgFS);
SAMF_var = nanstd(SAMFS);
ERGASF_var = nanstd(ERGASFS);
SCC_GTF_var = nanstd(SCC_GTFS);
QF_var = nanstd(QFS);

% [m1, v1, l1, r1] = normfit(QS, 0.05);
% fprintf('Q: %0.4f±%0.4f 置信区间：[%0.4f,%0.4f]\r\n',m1, v1,l1(1),l1(2));
% [m2, v2, l2, r2] = normfit(Q_avgS, 0.05);
% fprintf('Q_avg: %0.4f±%0.4f 置信区间：[%0.4f,%0.4f]\r\n',m2, v2, l2(1), l2(2));
% [m3, v3, l3, r3] = normfit(SAMS, 0.05);
% fprintf('SAM: %0.4f±%0.4f 置信区间：[%0.4f,%0.4f]\r\n',m3, v3, l3(1), l3(2));
% [m4, v4, l4, r4] = normfit(ERGASS, 0.05);
% fprintf('ERGAS: %0.4f±%0.4f 置信区间：[%0.4f,%0.4f]\r\n',m4, v4, l4(1), l4(2));
% [m5, v5, l5, r5] = normfit(SCC_GTS, 0.05);
% fprintf('SCC_GT: %0.4f±%0.4f 置信区间：[%0.4f,%0.4f]\r\n',m5, v5, l5(1), l5(2));
% 
% 
% [m7, v7, l7, r7] = normfit(D_lambdaS, 0.05);
% fprintf('D_lambda: %0.4f±%0.4f 置信区间：[%0.4f,%0.4f]\r\n',m7, v7, l7(1), l7(2));
% [m8, v8, l8, r8] = normfit(D_SS, 0.05);
% fprintf('D_S: %0.4f±%0.4f 置信区间：[%0.4f,%0.4f]\r\n',m8, v8, l8(1), l8(2));
% [m9, v9, l9, r9] = normfit(QNR_indexS, 0.05);
% fprintf('QNRI: %0.4f±%0.4f 置信区间：[%0.4f,%0.4f]\r\n',m9, v9, l9(1), l9(2));

fprintf('Q: %0.4f±%0.4f\n',Q_mean, Q_var);
fprintf('Q_avg: %0.4f±%0.4f\n',Q_avg_mean, Q_avg_var);
fprintf('SAM: %0.4f±%0.4f\n',SAM_mean, SAM_var);
fprintf('ERGAS: %0.4f±%0.4f\n',ERGAS_mean, ERGAS_var);
fprintf('SCC_GT: %0.4f±%0.4f\n',SCC_GT_mean, SCC_GT_var);
fprintf('D_lambda_s: %0.4f±%0.4f\n',dll_mean, dll_var);
fprintf('D_S_s: %0.4f±%0.4f\n',dss_mean, dss_var);
fprintf('QNRI_s: %0.4f±%0.4f\n',qnrs_mean, qnrs_var);    
    
fprintf('D_lambda: %0.4f±%0.4f\n',D_lambda_mean, D_lambda_var);
fprintf('D_S: %0.4f±%0.4f\n',D_S_mean, D_S_var);
fprintf('QNRI: %0.4f±%0.4f\n',QNR_index_mean, QNR_index_var);
fprintf('SAM_FULL: %0.4f±%0.4f\n',SAM_index_mean, SAM_index_var);
fprintf('SCC_FULL: %0.4f±%0.4f\n',sCC_mean, sCC_var);

fprintf('Q_FULL: %0.4f±%0.4f\n',QF_mean, QF_var);
fprintf('Q_avg_FULL: %0.4f±%0.4f\n',Q_avgF_mean, Q_avgF_var);
fprintf('SAM_FULL: %0.4f±%0.4f\n',SAMF_mean, SAMF_var);
fprintf('ERGAS_FULL: %0.4f±%0.4f\n',ERGASF_mean, ERGASF_var);
fprintf('SCC_GT_FULL: %0.4f±%0.4f\n',SCC_GTF_mean, SCC_GTF_var);

save(save_path, 'Q_avgS', 'SAMS', 'ERGASS', 'QS', 'SCC_GTS', 'D_lambdaS', 'D_SS', 'QNR_indexS');
