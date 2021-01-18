%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           Post-Processing (PP) fusion by exploiting the Lee et al. algorithm. 
% 
% Interface:
%           I_Fus_PP = Fusion_Procedure_MTF_GLP_HPM_PP(I_PAN,I_MS,I_PAN_LP,ratio)
%
% Inputs:
%           I_PAN:          PAN image;
%           I_MS:           MS image upsampled at PAN scale;
%           I_PAN_LP:       PAN image at MS spatial resolution; 
%           sensor:         String for type of sensor (e.g. 'WV2','IKONOS');
%           ratio:          Scale ratio between MS and PAN. Pre-condition: Integer value.
%
% Outputs:
%           I_Fus_PP:       Image after PP.
% 
% References:
%           [Lee10]         J. Lee and C. Lee, “Fast and efficient panchromatic sharpening,” IEEE Transactions on Geoscience and Remote Sensing, vol. 48, no. 1,
%                           pp. 155–163, January 2010.
%           [Vivone14]      G. Vivone, L. Alparone, J. Chanussot, M. Dalla Mura, A. Garzelli, G. Licciardi, R. Restaino, and L. Wald, “A Critical Comparison Among Pansharpening Algorithms”, 
%                           IEEE Transaction on Geoscience and Remote Sensing, 2014. (Accepted)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function I_Fus_PP = Fusion_Procedure_MTF_GLP_HPM_PP(I_PAN,I_MS,I_PAN_LP,ratio)

imageHR = double(I_PAN);
I_MS = double(I_MS);

%%%%%%%%%%% Step 1: Reconstruction

%%% Bilinear interpolation
I_MS_I = imresize(I_MS,ratio,'bilinear');

I_Fus = I_MS_I .* (imageHR ./ (I_PAN_LP + eps));

%%% Mean corrections
for ii = 1 : size(I_MS,3)
    h = I_MS(:,:,ii);
    hf = I_Fus(:,:,ii);
    I_Fus(:,:,ii) = I_Fus(:,:,ii) - mean(hf(:)) + mean(h(:));
end

%%%%%%%%%%% Step 2: PostProcessing

%%% Step 2.1: Edge Image
Im_Lap_Fus = zeros(size(I_Fus));
for idim = 1 : size(I_Fus,3)
    Im_Lap_Fus(:,:,idim)= imfilter(I_Fus(:,:,idim),fspecial('laplacian'));
end

%%% Step 2.2 and Step 2.3: Binary Edge Image and PP Fusion Image
I_Fus_PP = I_Fus;
for idim = 1 : size(Im_Lap_Fus,3)
    h = Im_Lap_Fus(:,:,idim);
    hP = imageHR(:,:,idim);
    hms = I_MS(:,:,idim);
    ind = find(h > 1.5 .* mean(hms(:)));
    hfusion = I_Fus_PP(:,:,idim);
    im_med = medfilt2(hP - hfusion);
    hfusion(ind) = hP(ind) - im_med(ind);
    I_Fus_PP(:,:,idim) = hfusion;
end

end