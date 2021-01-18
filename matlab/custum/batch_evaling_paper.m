% 批量融合影像保存测评
% 参数
ratio = 4;
Qblocks_size = 32;
L = 11;
flag_cut_bounds = 1;
dim_cut = 11;
thvalues = 0;
sensor = 'WV2';%'IKONOS';%'QB';%%'WV3';% %设置更换数据集 非常重要！！！！'none';%
im_tag = 'none';%'Tls1';%'none'; % 设置
% 需要处理的路径
test_dataset_path = 'D:\实验\影像融合\WashingtonDC_View-Ready_8_Band_Bundle_40cm\055675869040\dataset\test_dataset';%'D:/实验/融合数据集/SpaceNet挑选/test_matlab'; % 2 QuickBird 6 WorldView-3 1 IKONOS 3 GaoFen-1 4 WorldView-4 5 WorldView-2
% 结果保存路径
result_path = 'D:\实验\影像融合\WashingtonDC_View-Ready_8_Band_Bundle_40cm\055675869040\dataset\result';%'D:\实验\影像融合\IKONOS\train_dataset\result';%'D:/实验/PsCycleGAN/result';

% 下采样LR路径
test_ms_path = fullfile(test_dataset_path, 'LR');
% 下采样HR
test_pan_path = fullfile(test_dataset_path, 'HR');
% 下采样FR
test_gt_path = fullfile(test_dataset_path, 'FR');

% 原始MS
test_origin_ms_path = fullfile(test_dataset_path, 'MS');
% 原始PAN
test_origin_pan_path = fullfile(test_dataset_path, 'PAN');


% 下采样MS数据列表
test_ms_paths = dir(fullfile(test_ms_path, '*.tif'));
test_ms_names = {test_ms_paths.name};
% 下采样PAN数据列表
test_pan_paths = dir(fullfile(test_pan_path, '*.tif'));
test_pan_names = {test_pan_paths.name};
% GT数据列表
test_gt_paths = dir(fullfile(test_gt_path, '*.tif'));
test_gt_names = {test_gt_paths.name};

% 原始MS数据列表
test_origin_ms_paths = dir(fullfile(test_origin_ms_path, '*.tif'));
test_origin_ms_names = {test_origin_ms_paths.name};
% 原始PAN数据列表
test_origin_pan_paths = dir(fullfile(test_origin_pan_path, '*.tif'));
test_origin_pan_names = {test_origin_pan_paths.name};

names = [{'GSA'},{'PRACS'},{'ATWT'},...
        {'MTF-GLP-CBD'}];
nameLength = length(names);
name_paths = cell(nameLength);
name_o_paths = cell(nameLength);
% 准备保存文件夹
for i=1:nameLength
    target_path = fullfile(result_path, names(i));
    target_path = target_path{1};
    % 创建主目录
    if exist(target_path)==0   %该文件夹不存在，则直接创建
        mkdir(target_path);
    end
    target_reduce_path = fullfile(target_path, 'reduce');
    if exist(target_reduce_path)==0   %该文件夹不存在，则直接创建
        mkdir(target_reduce_path);
    end
    target_full_path = fullfile(target_path, 'full');
    if exist(target_full_path)==0   %该文件夹不存在，则直接创建
        mkdir(target_full_path);
    end
    name_paths(i) = {target_reduce_path};
    name_o_paths(i) = {target_full_path};
end
% 进度条
bar = waitbar(0,'准备中...');    % waitbar显示进度条
datalength = length(test_ms_names);
% 初始化评价指标数组
NumAlgs = 19;
NumIndexes = 6; % 加时间
MatrixResults = zeros(datalength,NumAlgs,NumIndexes);
ONumAlgs = 19;
ONumIndexes = 4; % 加时间
OMatrixResults = zeros(datalength,ONumAlgs,ONumIndexes);
DlS = zeros(1, datalength);
DSS = zeros(1, datalength);
% 遍历处理
for i=1:datalength
    
    % 下采样LR
    current_test_ms_path = fullfile(test_ms_path, test_ms_names(i));
    %将cell类型转换为string类型
    current_test_ms_path = current_test_ms_path{1};
    %读取LR数据
    I_MS_LR = imread(current_test_ms_path);
    % 上采样至PAN大小
    I_MS= zeros(round(size(I_MS_LR,1)*ratio),round(size(I_MS_LR,2)*ratio),size(I_MS_LR,3));
    for idim=1:size(I_MS_LR,3)
        I_MS(:,:,idim) = imresize(I_MS_LR(:,:,idim),ratio);
    end
    
    % 下采样HR
    current_test_pan_path = fullfile(test_pan_path, test_pan_names(i));
    %将cell类型转换为string类型
    current_test_pan_path = current_test_pan_path{1};
    %读取PAN数据
    I_PAN = imread(current_test_pan_path);
    
    % 下采样HR
    current_test_gt_path = fullfile(test_gt_path, test_gt_names(i));
    %将cell类型转换为string类型
    current_test_gt_path = current_test_gt_path{1};
    %读取融合数据
    I_GT = imread(current_test_gt_path);
    I_MS = double(I_MS);
    I_PAN = double(I_PAN);
    I_GT = double(I_GT);
  
    cd ..
    %% GSA
    cd GS
    t2=tic;
    I_GSA = GSA(I_MS,I_PAN,I_MS_LR,ratio);
    time_GSA = toc(t2);
    cd ..
    [Q_avg_GSA, SAM_GSA, ERGAS_GSA, SCC_GT_GSA, Q_GSA] = indexes_evaluation(I_GSA,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
    MatrixResults(i,1,:) = [Q_GSA,Q_avg_GSA,SAM_GSA,ERGAS_GSA,SCC_GT_GSA,time_GSA];
    % 保存
    cd custum
    path = fullfile(name_paths(1), test_ms_names(i));
    writeTiff(I_GSA, path{1});
    cd ..
    %% PRACS
    cd PRACS
    t2=tic;
    I_PRACS = PRACS(I_MS,I_PAN,ratio);
    time_PRACS = toc(t2);
    cd ..
    [Q_avg_PRACS, SAM_PRACS, ERGAS_PRACS, SCC_GT_PRACS, Q_PRACS] = indexes_evaluation(I_PRACS,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
    MatrixResults(i,2,:) = [Q_PRACS,Q_avg_PRACS,SAM_PRACS,ERGAS_PRACS,SCC_GT_PRACS,time_PRACS];
    % 保存
    cd custum
    path = fullfile(name_paths(2), test_ms_names(i));
    writeTiff(I_PRACS, path{1});
    cd ..
   %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% MultiResolution Analysis %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   
    %% ATWT
    cd Wavelet
    t2=tic;
    I_ATWT = ATWT(I_MS,I_PAN,ratio);
    time_ATWT = toc(t2);
    cd ..
    [Q_avg_ATWT, SAM_ATWT, ERGAS_ATWT, SCC_GT_ATWT, Q_ATWT] = indexes_evaluation(I_ATWT,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
    MatrixResults(i,3,:) = [Q_ATWT,Q_avg_ATWT,SAM_ATWT,ERGAS_ATWT,SCC_GT_ATWT,time_ATWT];
    % 保存
    cd custum
    path = fullfile(name_paths(3), test_ms_names(i));
    writeTiff(I_ATWT, path{1});
    cd ..
   
    %% MTF-GLP-CBD
    cd GS
    t2=tic;
    I_MTF_GLP_CBD = GS2_GLP(I_MS,I_PAN,ratio,sensor,im_tag);
    time_MTF_GLP_CBD = toc(t2);
    cd ..
    [Q_avg_MTF_GLP_CBD, SAM_MTF_GLP_CBD, ERGAS_MTF_GLP_CBD, SCC_GT_MTF_GLP_CBD, Q_MTF_GLP_CBD] = indexes_evaluation(I_MTF_GLP_CBD,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
    MatrixResults(i,4,:) = [Q_MTF_GLP_CBD,Q_avg_MTF_GLP_CBD,SAM_MTF_GLP_CBD,ERGAS_MTF_GLP_CBD,SCC_GT_MTF_GLP_CBD,time_MTF_GLP_CBD];
    % 保存
    cd custum
    path = fullfile(name_paths(4), test_ms_names(i));
    writeTiff(I_MTF_GLP_CBD, path{1});
    cd ..

    %%%%%%%%%%%%%%全
    % MS
    current_test_origin_ms_path = fullfile(test_origin_ms_path, test_origin_ms_names(i));
    %将cell类型转换为string类型
    current_test_origin_ms_path = current_test_origin_ms_path{1};
    %读取融合数据
    I_MS_LR = imread(current_test_origin_ms_path);

    % PAN
    current_test_origin_pan_path = fullfile(test_origin_pan_path, test_origin_pan_names(i));
    %将cell类型转换为string类型
    current_test_origin_pan_path = current_test_origin_pan_path{1};
    %读取融合数据
    I_PAN = imread(current_test_origin_pan_path);

    % 上采样
    I_MS = zeros(round(size(I_MS_LR,1)*ratio),round(size(I_MS_LR,2)*ratio),size(I_MS_LR,3));
    for idim=1:size(I_MS_LR,3)
        I_MS(:,:,idim) = imresize(I_MS_LR(:,:,idim),ratio);
    end
    
    I_MS = double(I_MS);
    I_PAN = double(I_PAN);
    I_MS_LR = double(I_MS_LR);
    
   
    %% GSA
    cd GS
    t2=tic;
    I_GSA = GSA(I_MS,I_PAN,I_MS_LR,ratio);
    time_GSA = toc(t2);
    cd ..
    [D_lambda_GSA,D_S_GSA,QNRI_GSA,SAM_GSA,SCC_GSA] = indexes_evaluation_FS(I_GSA,I_MS_LR,I_PAN,L,thvalues,I_MS,sensor,im_tag,ratio);
    OMatrixResults(i,1,:) = [D_lambda_GSA,D_S_GSA,QNRI_GSA,time_GSA];
    % 保存
    cd custum
    path = fullfile(name_o_paths(1), test_ms_names(i));
    writeTiff(I_GSA, path{1});
    cd ..
    %% PRACS
    cd PRACS
    t2=tic;
    I_PRACS = PRACS(I_MS,I_PAN,ratio);
    time_PRACS = toc(t2);
    cd ..
    [D_lambda_PRACS,D_S_PRACS,QNRI_PRACS,SAM_PRACS,SCC_PRACS] = indexes_evaluation_FS(I_PRACS,I_MS_LR,I_PAN,L,thvalues,I_MS,sensor,im_tag,ratio);
    OMatrixResults(i,2,:) = [D_lambda_PRACS,D_S_PRACS,QNRI_PRACS,time_PRACS];
    % 保存
    cd custum
    path = fullfile(name_o_paths(2), test_ms_names(i));
    writeTiff(I_PRACS, path{1});
    cd ..
    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% MultiResolution Analysis %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   
    %% ATWT
    cd Wavelet
    t2=tic;
    I_ATWT = ATWT(I_MS,I_PAN,ratio);
    time_ATWT = toc(t2);
    cd ..
    [D_lambda_ATWT,D_S_ATWT,QNRI_ATWT,SAM_ATWT,SCC_ATWT] = indexes_evaluation_FS(I_ATWT,I_MS_LR,I_PAN,L,thvalues,I_MS,sensor,im_tag,ratio);
    OMatrixResults(i,3,:) = [D_lambda_ATWT,D_S_ATWT,QNRI_ATWT,time_ATWT];
    % 保存
    cd custum
    path = fullfile(name_o_paths(3), test_ms_names(i));
    writeTiff(I_ATWT, path{1});
    cd ..
    
    %% MTF-GLP-CBD
    cd GS
    t2=tic;
    I_MTF_GLP_CBD = GS2_GLP(I_MS,I_PAN,ratio,sensor,im_tag);
    time_MTF_GLP_CBD = toc(t2);
    cd ..
    [D_lambda_MTF_GLP_CBD,D_S_MTF_GLP_CBD,QNRI_MTF_GLP_CBD,SAM_MTF_GLP_CBD,SCC_MTF_GLP_CBD] = indexes_evaluation_FS(I_MTF_GLP_CBD,I_MS_LR,I_PAN,L,thvalues,I_MS,sensor,im_tag,ratio);
    OMatrixResults(i,4,:) = [D_lambda_MTF_GLP_CBD,D_S_MTF_GLP_CBD,QNRI_MTF_GLP_CBD,time_MTF_GLP_CBD];
    % 保存
    cd custum
    path = fullfile(name_o_paths(4), test_ms_names(i));
    writeTiff(I_MTF_GLP_CBD, path{1});
    cd ..
    cd custum
    % 更新进度条
    str=['处理中...',num2str(100*i/datalength),'%'];    % 百分比形式显示处理进程,不需要删掉这行代码就行
    waitbar(i/datalength,bar,str)                      % 不注释显示
end

reduce_mean = nanmean(MatrixResults, 1);
reduce_var = nanstd(MatrixResults, 1);
full_mean = nanmean(OMatrixResults, 1);
full_var = nanstd(OMatrixResults, 1);
%OMatrixResults = OMatrixResults(~isnan(OMatrixResults)); % 移除NAN

% for i=1:nameLength
%     name = names(i);
%     fprintf('%s\n',name{1});
%     [m1, v1, l1, r1] = normfit(MatrixResults(:,i,1), 0.05);
%     fprintf('Q: %0.4f±%0.4f 置信区间：[%0.4f,%0.4f]\n',m1, v1,l1(1),l1(2));
%     [m2, v2, l2, r2] = normfit(MatrixResults(:,i,2), 0.05);
%     fprintf('Q_avg: %0.4f±%0.4f 置信区间：[%0.4f,%0.4f]\n',m2, v2, l2(1), l2(2));
%     [m3, v3, l3, r3] = normfit(MatrixResults(:,i,3), 0.05);
%     fprintf('SAM: %0.4f±%0.4f 置信区间：[%0.4f,%0.4f]\n',m3, v3, l3(1), l3(2));
%     [m4, v4, l4, r4] = normfit(MatrixResults(:,i,4), 0.05);
%     fprintf('ERGAS: %0.4f±%0.4f 置信区间：[%0.4f,%0.4f]\n',m4, v4, l4(1), l4(2));
%     [m5, v5, l5, r5] = normfit(MatrixResults(:,i,5), 0.05);
%     fprintf('SCC_GT: %0.4f±%0.4f 置信区间：[%0.4f,%0.4f]\n',m5, v5, l5(1), l5(2));
%     [m6, v6, l6, r6] = normfit(MatrixResults(:,i,6), 0.05);
%     fprintf('reduce_time: %0.4f±%0.4f 置信区间：[%0.4f,%0.4f]\n',m6, v6, l6(1), l6(2));
%     
%     [m7, v7, l7, r7] = normfit(OMatrixResults(:,i,1), 0.05);
%     fprintf('D_lambda: %0.4f±%0.4f 置信区间：[%0.4f,%0.4f]\n',m7, v7, l7(1), l7(2));
%     [m8, v8, l8, r8] = normfit(OMatrixResults(:,i,2), 0.05);
%     fprintf('D_S: %0.4f±%0.4f 置信区间：[%0.4f,%0.4f]\n',m8, v8, l8(1), l8(2));
%     [m9, v9, l9, r9] = normfit(OMatrixResults(:,i,3), 0.05);
%     fprintf('QNRI: %0.4f±%0.4f 置信区间：[%0.4f,%0.4f]\n',m9, v9, l9(1), l9(2));
%     [m10, v10, l10, r10] = normfit(OMatrixResults(:,i,4), 0.05);
%     fprintf('full_time: %0.4f±%0.4f 置信区间：[%0.4f,%0.4f]\n',m10, v10, l10(1), l10(2));
%     
% end
    
for i=1:nameLength
    name = names(i);
    fprintf('%s\n',name{1});
    fprintf('Q: %0.4f±%0.4f\n',reduce_mean(:,i,1), reduce_var(:,i,1));
    fprintf('Q_avg: %0.4f±%0.4f\n',reduce_mean(:,i,2), reduce_var(:,i,2));
    fprintf('SAM: %0.4f±%0.4f\n',reduce_mean(:,i,3), reduce_var(:,i,3));
    fprintf('ERGAS: %0.4f±%0.4f\n',reduce_mean(:,i,4), reduce_var(:,i,4));
    fprintf('SCC_GT: %0.4f±%0.4f\n',reduce_mean(:,i,5), reduce_var(:,i,5));
    
    fprintf('reduce_time: %0.4f±%0.4f\n',reduce_mean(:,i,6), reduce_var(:,i,6));
    fprintf('D_lambda: %0.4f±%0.4f\n',full_mean(:,i,1), full_var(:,i,1));
    fprintf('D_S: %0.4f±%0.4f\n',full_mean(:,i,2), full_var(:,i,2));
    fprintf('QNRI: %0.4f±%0.4f\n',full_mean(:,i,3), full_var(:,i,3));
    fprintf('full_time: %0.4f±%0.4f\n',full_mean(:,i,4), full_var(:,i,4));
end