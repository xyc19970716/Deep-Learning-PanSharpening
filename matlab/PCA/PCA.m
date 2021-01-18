%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           PCA fuses the upsampled MultiSpectral (MS) and PANchromatic (PAN) images by 
%           exploiting the Principal Component Analysis (PCA) transformation. 
% 
% Interface:
%           I_Fus_PCA = PCA(I_MS,I_PAN)
%
% Inputs:
%           I_MS:           MS image upsampled at PAN scale;
%           I_PAN:          PAN image.
%
% Outputs:
%           I_Fus_PCA:      PCA pansharpened image.
% 
% References:
%           [Chavez89]      P. S. Chavez Jr. and A. W. Kwarteng, “Extracting spectral contrast in Landsat Thematic Mapper image data using selective principal
%                           component analysis,? Photogrammetric Engineering and Remote Sensing, vol. 55, no. 3, pp. 339?348, March 1989.
%           [Vivone14]      G. Vivone, L. Alparone, J. Chanussot, M. Dalla Mura, A. Garzelli, G. Licciardi, R. Restaino, and L. Wald, “A Critical Comparison Among Pansharpening Algorithms?, 
%                           IEEE Transaction on Geoscience and Remote Sensing, 2014. (Accepted)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function I_Fus_PCA = PCA(I_MS,I_PAN)

imageLR = double(I_MS);
imageHR = double(I_PAN);

[n,m,d]=size(imageLR);
imageLR = reshape(imageLR, [n*m,d]);

% PCA transform on MS bands
[W,PCAData] = princomp(imageLR);%[W,PCAData] = pca(imageLR);

F = reshape(PCAData, [n,m,d]); 

% Equalization
I = F(:,:,1);
imageHR = (imageHR - mean(imageHR(:)))*std2(I)/std(imageHR(:)) + mean2(I);

% Replace 1st band with PAN
F(:,:,1) = imageHR;

% Inverse PCA
I_Fus_PCA = reshape(F,[n*m,d]) * W';
I_Fus_PCA = reshape(I_Fus_PCA, [n,m,d]);

% Final Linear Equalization
for ii = 1 : size(I_MS,3)
    h = I_Fus_PCA(:,:,ii);
    I_Fus_PCA(:,:,ii) = h - mean2(h) + mean2(squeeze(double(I_MS(:,:,ii))));
end

end

