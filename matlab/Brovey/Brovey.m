%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           Brovey fuses the upsampled MultiSpectral (MS) and PANchromatic (PAN) images by 
%           exploiting the Brovey pansharpening algorithm. 
% 
% Interface:
%           I_Fus_Brovey = Brovey(I_MS,I_PAN)
%
% Inputs:
%           I_MS:         MS image upsampled at PAN scale;
%           I_PAN:        PAN image.
%
% Outputs:
%           I_Fus_Brovey: Brovey pasharpened image.
% 
% References:
%           [Gillespie87] A. R. Gillespie, A. B. Kahle, and R. E. Walker, “Color enhancement of highly correlated images-II. Channel ratio and “Chromaticity”
%                         Transform techniques,” Remote Sensing of Environment, vol. 22, no. 3, pp. 343–365, August 1987.
%           [Tu01]        T.-M. Tu, S.-C. Su, H.-C. Shyu, and P. S. Huang, “A new look at IHS-like image fusion methods,” Information Fusion, vol. 2, no. 3,
%                         pp. 177–186, September 2001.
%           [Vivone14]    G. Vivone, L. Alparone, J. Chanussot, M. Dalla Mura, A. Garzelli, G. Licciardi, R. Restaino, and L. Wald, “A Critical Comparison Among Pansharpening Algorithms”, 
%                         IEEE Transaction on Geoscience and Remote Sensing, 2014. (Accepted)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function I_Fus_Brovey = Brovey(I_MS,I_PAN)

imageLR = double(I_MS);
imageHR = double(I_PAN);

% Intensity Component
I = mean(imageLR,3);

% Equalization PAN component
imageHR = (imageHR - mean2(imageHR)).*(std2(I)./std2(imageHR)) + mean2(I);  

I_Fus_Brovey = imageLR .* repmat(imageHR./(I+eps),[1 1 size(imageLR,3)]);

end