% �²������ݼ�
% MS��PAN��������
ratio = 4;%4;
% ����
sensor = 'GeoEye1';%'WV2';%'IKONOS';%'QB';%'WV3';%'none';%%%'WV3';%

% ��Ҫ�����·��
dataset_path = 'H:\����У��\�ں�\dataset\train_dataset';%'D:\ʵ��\Ӱ���ں�\data\NBU_PansharpRSData\1 Satellite_Dataset\Dataset\6 WorldView-3\train_dataset';%D:/ʵ��/�ں����ݼ�/SpaceNet��ѡ/test_dataset'; 1 IKONOS 6 WorldView-3 2 QuickBird 3 GaoFen-1 4 WorldView-4 5 WorldView-2
% ԭʼMS���·��
dataset_ms_path = fullfile(dataset_path, 'MS');
% ԭʼPAN���·��
dataset_pan_path = fullfile(dataset_path, 'PAN');
% �²���LR
target_ms_path = fullfile(dataset_path, 'LR');
if exist(target_ms_path)==0   %���ļ��в����ڣ���ֱ�Ӵ���
    mkdir(target_ms_path);
end

% �²���HR
target_pan_path = fullfile(dataset_path, 'HR');
if exist(target_pan_path)==0   %���ļ��в����ڣ���ֱ�Ӵ���
    mkdir(target_pan_path);
end
% �²���FR
target_gt_path = fullfile(dataset_path, 'FR');
if exist(target_gt_path)==0   %���ļ��в����ڣ���ֱ�Ӵ���
    mkdir(target_gt_path);
end
% ԭʼMS�����б�
dataset_ms_paths = dir(fullfile(dataset_ms_path, '*.tif'));
dataset_ms_names = {dataset_ms_paths.name};
% ԭʼPAN�����б�
dataset_pan_paths = dir(fullfile(dataset_pan_path, '*.tif'));
dataset_pan_names = {dataset_pan_paths.name};

% ������
bar = waitbar(0,'׼����...');    % waitbar��ʾ������
datalength = length(dataset_ms_names);
% ��������
for i=1:length(dataset_ms_names)
    % ��Ҫ����MS·��
    current_ms_path = fullfile(dataset_ms_path, dataset_ms_names(i));
    %��cell����ת��Ϊstring����
    current_ms_path = current_ms_path{1};
    %��ȡMS����
    ms_data = imread(current_ms_path);
    % ��Ҫ����PAN·��
    current_pan_path = fullfile(dataset_pan_path, dataset_pan_names(i));
    %��cell����ת��Ϊstring����
    current_pan_path = current_pan_path{1};
    %��ȡPAN����
    pan_data = imread(current_pan_path);
    % �²���MS��PAN
    cd ..
    [ms_down_data, pan_down_data] = resize_images(ms_data,pan_data,ratio,sensor);
    cd custum
    % ����LR
    ms_path = fullfile(target_ms_path, dataset_ms_names(i));
    ms_path = ms_path{1};
    writeTiff(ms_down_data, ms_path);
    % ����HR
    pan_path = fullfile(target_pan_path, dataset_pan_names(i));
    pan_path = pan_path{1};
    writeTiff(pan_down_data, pan_path);
    % ����FR
    gt_path = fullfile(target_gt_path, dataset_ms_names(i));
    gt_path = gt_path{1};
    writeTiff(ms_data, gt_path);
    % ���½�����
    str=['������...',num2str(100*i/datalength),'%'];    % �ٷֱ���ʽ��ʾ�������,����Ҫɾ�����д������
    waitbar(i/datalength,bar,str)                      % ��ע����ʾ
end
    