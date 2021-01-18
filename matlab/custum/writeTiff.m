function writeTiff(imgData, imgPath)
    t = Tiff(imgPath,'w');
    % Ӱ���С��Ϣ��������Ƚϼ򵥣�
    tagstruct.ImageLength = size(imgData,1); % Ӱ��ĳ���
    tagstruct.ImageWidth = size(imgData,2);  % Ӱ��Ŀ��

    % ��ɫ�ռ���ͷ�ʽ����ϸ������3.1��
    tagstruct.Photometric = 1;

    % ÿ�����ص���ֵλ����singleΪ�����ȸ����ͣ�����32ΪϵͳΪ32
    tagstruct.BitsPerSample = 16;
    % ÿ�����صĲ��θ�����һ��ͼ��Ϊ1��3�����Ƕ���ң��Ӱ����ڶ���������Գ�������3
    tagstruct.SamplesPerPixel = size(imgData,3);
    tagstruct.RowsPerStrip = 32;
    tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
    % ��ʾ����Ӱ������
    tagstruct.Software = 'MATLAB'; 
   
    % ����Tiff�����tag
    t.setTag(tagstruct);

    % ��׼����ͷ�ļ�����ʼд����
    t.write(uint16(imgData));
    % �ر�Ӱ��
    t.close;
end