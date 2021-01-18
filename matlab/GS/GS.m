%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           GS fuses the upsampled MultiSpectral (MS) and PANchromatic (PAN) images by 
%           exploiting the Gram-Schmidt (GS) transformation.
% 
% Interface:
%           I_Fus_GS = GS(I_MS,I_PAN)
%
% Inputs:
%           I_MS:       MS image upsampled at PAN scale;
%           I_PAN:      PAN image.
%
% Outputs:
%           I_Fus_GS:   GS pasharpened image.
% 
% References:
%           [Laben00]   C. A. Laben and B. V. Brower, “Process for enhancing the spatial resolution of multispectral imagery using pan-sharpening,” Eastman
%                       Kodak Company, Tech. Rep. US Patent # 6,011,875, 2000.
%           [Aiazzi07]  B. Aiazzi, S. Baronti, and M. Selva, “Improving component substitution Pansharpening through multivariate regression of MS+Pan
%                       data,” IEEE Transactions on Geoscience and Remote Sensing, vol. 45, no. 10, pp. 3230–3239, October 2007.
%           [Vivone14]  G. Vivone, L. Alparone, J. Chanussot, M. Dalla Mura, A. Garzelli, G. Licciardi, R. Restaino, and L. Wald, “A Critical Comparison Among Pansharpening Algorithms”, 
%                       IEEE Transaction on Geoscience and Remote Sensing, 2014. (Accepted)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function I_Fus_GS = GS(I_MS,I_PAN)

imageLR = double(I_MS);
imageHR = double(I_PAN);

%%% Remove means from imageLR
imageLR0 = zeros(size(I_MS));
for ii = 1 : size(I_MS,3), imageLR0(:,:,ii) = imageLR(:,:,ii) - mean2(imageLR(:,:,ii)); end

%%% Intensity
I = mean(imageLR,3); 

%%% Remove mean from I
I0 = I - mean2(I);

imageHR = (imageHR - mean2(imageHR)) .* (std2(I0)./std2(imageHR)) + mean2(I0);

%%% Coefficients
g = ones(1,1,size(I_MS,3)+1);
for ii = 1 : size(I_MS,3)
    h = imageLR0(:,:,ii);
    c = cov(I0(:),h(:));
    g(1,1,ii+1) = c(1,2)/var(I0(:));
end

%%% Detail Extraction
delta = imageHR - I0;
deltam = repmat(delta(:),[1 size(I_MS,3)+1]);

%%% Fusion
V = I0(:);
for ii = 1 : size(I_MS,3)
    h = imageLR0(:,:,ii);
    V = cat(2,V,h(:));
end

gm = zeros(size(V));
for ii = 1 : size(g,3)
    gm(:,ii) = squeeze(g(1,1,ii)) .* ones(size(I_MS,1).*size(I_MS,2),1);
end

V_hat = V + deltam .* gm;

%%% Reshape fusion result
I_Fus_GS = reshape(V_hat(:,2:end),[size(I_MS,1) size(I_MS,2) size(I_MS,3)]);

% Final Mean Equalization
for ii = 1 : size(I_MS,3)
    h = I_Fus_GS(:,:,ii);
    I_Fus_GS(:,:,ii) = h - mean2(h) + mean2(imageLR(:,:,ii));
end

end