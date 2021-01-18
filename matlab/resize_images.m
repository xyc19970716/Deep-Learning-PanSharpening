%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           Generate the low resolution PANchromatic (PAN) and MultiSpectral (MS) images according to the Wald's protocol. 
%           
% Interface:
%           [I_MS_LR, I_PAN_LR] = resize_images(I_MS,I_PAN,ratio,sensor)
%
% Inputs:
%           I_MS:           MS image upsampled at PAN scale;
%           I_PAN:          PAN image;
%           ratio:          Scale ratio between MS and PAN. Pre-condition: Integer value;
%           sensor:         String for type of sensor (e.g. 'WV2', 'IKONOS').
%
% Outputs:
%           I_MS_LR:        Low Resolution MS image;
%           I_PAN_LR:       Low Resolution PAN image.
% 
% References:
%           [Wald97]        L. Wald, T. Ranchin, and M. Mangolini, “Fusion of satellite images of different spatial resolutions: assessing the quality of resulting images,?
%                           Photogrammetric Engineering and Remote Sensing, vol. 63, no. 6, pp. 691?99, June 1997.
%           [Aiazzi06]      B. Aiazzi, L. Alparone, S. Baronti, A. Garzelli, and M. Selva, “MTF-tailored multiscale fusion of high-resolution MS and Pan imagery,?
%                           Photogrammetric Engineering and Remote Sensing, vol. 72, no. 5, pp. 591?96, May 2006.
%           [Vivone14]      G. Vivone, L. Alparone, J. Chanussot, M. Dalla Mura, A. Garzelli, G. Licciardi, R. Restaino, and L. Wald, “A Critical Comparison Among Pansharpening Algorithms? 
%                           IEEE Transaction on Geoscience and Remote Sensing, 2014. (Accepted)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [I_MS_LR, I_PAN_LR] = resize_images(I_MS,I_PAN,ratio,sensor)

I_MS = double(I_MS);
I_PAN = double(I_PAN);
flag_PAN_MTF = 0;

switch sensor
    case 'QB' 
        flag_resize_new = 2; % MTF usage
        GNyq = [0.34 0.32 0.30 0.22]; % Band Order: B,G,R,NIR
        GNyqPan = 0.15;
    case 'IKONOS'
        flag_resize_new = 2; % MTF usage
        GNyq = [0.26,0.28,0.29,0.28]; % Band Order: B,G,R,NIR
        GNyqPan = 0.17;
    case 'GeoEye1' 
        flag_resize_new = 2; % MTF usage
        GNyq = [0.23,0.23,0.23,0.23]; % Band Order: B,G,R,NIR
        GNyqPan = 0.16;
    case 'WV2'
        flag_resize_new = 2; % MTF usage
        GNyq = [0.35 .* ones(1,7), 0.27];%GNyq = [0.35, 0.35, 0.35, 0.35];% 
        GNyqPan = 0.11;
    case 'WV3'
        flag_resize_new = 2; % MTF usage
        GNyq = [0.355, 0.360,0.365,0.335];%GNyq = [0.325, 0.355, 0.360, 0.350, 0.365, 0.360, 0.335, 0.315];
        GNyqPan = 0.5;  
    case 'none'
        flag_resize_new = 1; % Bicubic Interpolator
end

if flag_resize_new == 1
    
    %%% Bicubic Interpolator MS
    I_MS_LP = zeros(round(size(I_MS,1)/ratio),round(size(I_MS,2)/ratio),size(I_MS,3));
    
    for idim=1:size(I_MS,3)
        I_MS_LP(:,:,idim) = imresize(I_MS(:,:,idim),1/ratio);
    end
    
    I_MS_LR = double(I_MS_LP);
    
    %%% Bicubic Interpolator PAN
    I_PAN_LR = imresize(I_PAN,1/ratio);
    
elseif flag_resize_new == 2
    
    %%% MTF
    
    %%% Filtering with sensor MTF MS
    N = 41;
    I_MS_LP = zeros(size(I_MS));
    fcut = 1 / ratio;
    
    for ii = 1 : size(I_MS,3)
        alpha = sqrt(((N-1)*(fcut/2))^2/(-2*log(GNyq(ii))));
        H = fspecial('gaussian', N, alpha);
        Hd = H./max(H(:));
        h = fwind1(Hd,kaiser(N));
        I_MS_LP(:,:,ii) = imfilter(I_MS(:,:,ii),real(h),'replicate');
    end
    
    if flag_PAN_MTF == 1
        %%% Filtering with sensor MTF PAN
        alpha = sqrt(((N-1)*(fcut/2))^2/(-2*log(GNyqPan)));
        H = fspecial('gaussian', N, alpha);
        Hd = H./max(H(:));
        h = fwind1(Hd,kaiser(N));
        I_PAN = imfilter(I_PAN,real(h),'replicate');
        %%% Decimation PAN
        I_PAN_LR = imresize(I_PAN,1/ratio,'nearest');
    else
        I_PAN_LR = imresize(I_PAN,1/ratio);
    end

    %%% Decimation MS
    I_MS_LR = imresize(I_MS_LP,1/ratio,'nearest');
        
end

end