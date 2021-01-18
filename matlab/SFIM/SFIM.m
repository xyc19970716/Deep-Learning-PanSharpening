%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           SFIM fuses the upsampled MultiSpectral (MS) and PANchromatic (PAN) images by 
%           exploiting the Smoothing Filter-based Intensity Modulation (SFIM) algorithm. 
% 
% Interface:
%           I_Fus_SFIM = SFIM(I_MS,I_PAN,ratio)
%
% Inputs:
%           I_MS:           MS image upsampled at PAN scale;
%           I_PAN:          PAN image;
%           ratio:          Scale ratio between MS and PAN. Pre-condition: Integer value.
%
% Outputs:
%           I_Fus_SFIM:     SFIM pansharpened image.
% 
% References:
%           [Liu00]         J. Liu, “Smoothing filter based intensity modulation: a spectral preserve image fusion technique for improving spatial details,”
%                           International Journal of Remote Sensing, vol. 21, no. 18, pp. 3461–3472, December 2000.
%           [Vivone14]      G. Vivone, L. Alparone, J. Chanussot, M. Dalla Mura, A. Garzelli, G. Licciardi, R. Restaino, and L. Wald, “A Critical Comparison Among Pansharpening Algorithms”, 
%                           IEEE Transaction on Geoscience and Remote Sensing, 2014. (Accepted)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function I_Fus_SFIM = SFIM(I_MS,I_PAN,ratio)

if ~ rem(ratio,2)
    ratio = ratio + 1;
end

I_PAN = repmat(I_PAN,[1 1 size(I_MS,3)]);

for ii = 1 : size(I_MS,3)    
  I_PAN(:,:,ii) = (I_PAN(:,:,ii) - mean2(I_PAN(:,:,ii))).*(std2(I_MS(:,:,ii))./std2(I_PAN(:,:,ii))) + mean2(I_MS(:,:,ii));  
end

[Height,Width,Bands]=size(I_MS);
I_Fus_SFIM=zeros(Height,Width,Bands,'double');

for i=1:Bands
    LRPan = imfilter(I_PAN(:,:,i),fspecial('average',[ratio ratio]),'replicate');
    I_Fus_SFIM(:,:,i)=I_MS(:,:,i).*I_PAN(:,:,i)./(LRPan+eps);
end

end

