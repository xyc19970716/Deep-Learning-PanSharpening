%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           Full resolution quality indexes. 
% 
% Interface:
%           [D_lambda,D_S,QNR_index,SAM_index,sCC] = indexes_evaluation_FS(I_F,I_MS_LR,I_PAN,L,th_values,I_MS,sensor,tag,ratio)
%
% Inputs:
%           I_F:                Fused Image;
%           I_MS_LR:            MS image;
%           I_PAN:              Panchromatic image;
%           L:                  Image radiometric resolution; 
%           th_values:          Flag. If th_values == 1, apply an hard threshold to the dynamic range;
%           I_MS:               MS image upsampled to the PAN size;
%           sensor:             String for type of sensor (e.g. 'WV2','IKONOS');
%           tag:                Image tag. Often equal to the field sensor. It makes sense when sensor is 'none'. It indicates the band number;
%           ratio:              Scale ratio between MS and PAN. Pre-condition: Integer value.
%
% Outputs:
%           D_lambda:           D_lambda index;
%           D_S:                D_S index;
%           QNR_index:          QNR index;
%           SAM_index:          Spectral Angle Mapper (SAM) index between fused and MS image;
%           sCC:                spatial Correlation Coefficient between fused and PAN images.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [D_lambda,D_S,QNR_index,SAM_index,sCC] = indexes_evaluation_FS(I_F,I_MS_LR,I_PAN,L,th_values,I_MS,sensor,tag,ratio)

if th_values
    I_F(I_F > 2^L) = 2^L;
    I_F(I_F < 0) = 0;
end

cd Quality_Indices

[QNR_index,D_lambda,D_S]= QNR(I_F,I_MS,I_PAN,sensor,ratio);

I_PAN = repmat(I_PAN,[1 1 size(I_MS,3)]);

Im_Lap_PAN = zeros(size(I_PAN));
for idim=1:size(I_PAN,3),
    Im_Lap_PAN(:,:,idim)= imfilter(I_PAN(:,:,idim),fspecial('sobel'));
end

I_F_D = MTF(I_F,sensor,tag,ratio);

SAM_index = SAM(I_MS_LR,I_F_D);

Im_Lap_F = zeros(size(I_F));
for idim=1:size(I_MS_LR,3),
    Im_Lap_F(:,:,idim)= imfilter(I_F(:,:,idim),fspecial('sobel'));
end

sCC=sum(sum(sum(Im_Lap_PAN.*Im_Lap_F)));
sCC=sCC/sqrt(sum(sum(sum((Im_Lap_PAN.^2)))));
sCC=sCC/sqrt(sum(sum(sum((Im_Lap_F.^2)))));

cd ..

end