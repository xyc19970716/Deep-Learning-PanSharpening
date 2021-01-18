%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           MTF_GLP_HPM_PP fuses the upsampled MultiSpectral (MS) and PANchromatic (PAN) images by 
%           exploiting the Lee et al. algorithm based on Modulation Transfer Function -
%           Generalized Laplacian Pyramid (MTF-GLP), High Pass Modulation (HPM), and Post-Processing (PP). 
% 
% Interface:
%           I_Fus_MTF_GLP_HPM_PP = MTF_GLP_HPM_PP(I_PAN,I_MS,sensor,tag,ratio)
%
% Inputs:
%           I_PAN:                  PAN image;
%           I_MS:                   MS image upsampled at PAN scale;
%           sensor:                 String for type of sensor (e.g. 'WV2','IKONOS');
%           tag:                    Image tag. Often equal to the field sensor. It makes sense when sensor is 'none'. It indicates the band number
%                                   in the latter case;
%           ratio:                  Scale ratio between MS and PAN. Pre-condition: Integer value.
%
% Outputs:
%           I_Fus_MTF_GLP_HPM_PP:   MTF_GLP_HPM_PP pansharpened image.
% 
% References:
%           [Lee10]                 J. Lee and C. Lee, “Fast and efficient panchromatic sharpening,?IEEE Transactions on Geoscience and Remote Sensing, vol. 48, no. 1,
%                                   pp. 155?63, January 2010.
%           [Vivone14]              G. Vivone, L. Alparone, J. Chanussot, M. Dalla Mura, A. Garzelli, G. Licciardi, R. Restaino, and L. Wald, “A Critical Comparison Among Pansharpening Algorithms? 
%                                   IEEE Transaction on Geoscience and Remote Sensing, 2014. (Accepted)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function I_Fus_MTF_GLP_HPM_PP = MTF_GLP_HPM_PP(I_PAN,I_MS,sensor,tag,ratio)

imageHR = double(I_PAN);
I_MS = double(I_MS);

imageHR = repmat(imageHR,[1 1 size(I_MS,3)]);
for ii = 1 : size(I_MS,3)    
  imageHR(:,:,ii) = (imageHR(:,:,ii) - mean2(imageHR(:,:,ii))).*(std2(I_MS(:,:,ii))./std2(imageHR(:,:,ii))) + mean2(I_MS(:,:,ii));  
end

switch sensor
    case 'QB' 
        GNyq = [0.34 0.32 0.30 0.22]; % Band Order: B,G,R,NIR
    case 'IKONOS'
        GNyq = [0.26,0.28,0.29,0.28]; % Band Order: B,G,R,NIR
    case 'GeoEye1'
        GNyq = [0.23,0.23,0.23,0.23]; % Band Order: B,G,R,NIR
    case 'WV2'
        GNyq = [0.35, 0.35,0.35,0.35];% GNyq = [0.35 .* ones(1,7), 0.27];
    case 'WV3'
        GNyq = [ 0.355, 0.360,0.365,0.335];%GNyq = [0.325, 0.355, 0.360, 0.350, 0.365, 0.360, 0.335, 0.315];
    case 'none'
        if strcmp(tag,'WV2')
            GNyq = 0.15 .* ones(1,8);
        else
            GNyq = 0.29 .* ones(1,size(I_MS,3));
        end
end

%%% LPF MTF
N = 41;
PAN_LP = zeros(size(imageHR));
nBands = size(I_MS,3);
fcut = 1/(ratio/2);
   
for ii = 1 : nBands
    alpha = sqrt((N*(fcut/2))^2/(-2*log(GNyq(ii))));
    H = fspecial('gaussian', N, alpha);
    Hd = H./max(H(:));
    h = fwind1(Hd,kaiser(N));
    PAN_LP(:,:,ii) = imfilter(imageHR(:,:,ii),real(h),'replicate');
end
PAN_LP = double(PAN_LP);

%%% Downsampling and Interpolation (h_{LPF})
PAN_LP_D = imresize(PAN_LP,1/(ratio/2),'nearest');

PAN_LP = imresize(PAN_LP_D,(ratio/2),'bilinear');

PAN_LP_D = imresize(PAN_LP,1/(ratio/2),'nearest');


%%% LPF MTF
N = 41;
PAN_LP_LP = zeros(size(PAN_LP_D));
nBands = size(I_MS,3);
fcut = 1/(ratio/2);
   
for ii = 1 : nBands
    alpha = sqrt((N*(fcut/2))^2/(-2*log(GNyq(ii))));
    H = fspecial('gaussian', N, alpha);
    Hd = H./max(H(:));
    h = fwind1(Hd,kaiser(N));
    PAN_LP_LP(:,:,ii) = imfilter(PAN_LP_D(:,:,ii),real(h),'replicate');
end

%%% Downsampling and Interpolation (h_{LPF})
PAN_LP_LP = imresize(PAN_LP_LP,1/(ratio/2),'nearest');

PAN_LP_LP = imresize(PAN_LP_LP,(ratio/2),'bilinear');

I_Fus_MTF_GLP_HPM_PP = Fusion_Procedure_MTF_GLP_HPM_PP(PAN_LP_D,I_MS,PAN_LP_LP,ratio/2);
I_Fus_MTF_GLP_HPM_PP = Fusion_Procedure_MTF_GLP_HPM_PP(imageHR,I_Fus_MTF_GLP_HPM_PP,PAN_LP,ratio/2);

end