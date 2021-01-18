%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           upsampling7tap interpolates the image I_Interpolated using a polynomial with 7 coefficients interpolator. 
% 
% Interface:
%           I_Interpolated = upsampling7tap(I_Interpolated,ratio)
%
% Inputs:
%           I_Interpolated: Image to interpolate;
%           ratio:          Scale ratio between MS and PAN. Pre-condition: Resize factors power of 2.
%
% Outputs:
%           I_Interpolated: Interpolated image.
% 
% References:
%           [Khan08]        M. M. Khan, J. Chanussot, L. Condat, and A. Montavert, “Indusion: Fusion of multispectral and panchromatic images using the
%                           induction scaling technique,” IEEE Geoscience and Remote Sensing Letters, vol. 5, no. 1, pp. 98–102, January 2008.
%           [Aiazzi02]      B. Aiazzi, L. Alparone, S. Baronti, and A. Garzelli, “Context-driven fusion of high spatial and spectral resolution images based on
%                           oversampled multiresolution analysis,” IEEE Transactions on Geoscience and Remote Sensing, vol. 40, no. 10, pp. 2300–2312, October
%                           2002.
%           [Aiazzi13]      B. Aiazzi, S. Baronti, M. Selva, and L. Alparone, “Bi-cubic interpolation for shift-free pan-sharpening,” ISPRS Journal of Photogrammetry
%                           and Remote Sensing, vol. 86, no. 6, pp. 65–76, December 2013.
%           [Vivone14]      G. Vivone, L. Alparone, J. Chanussot, M. Dalla Mura, A. Garzelli, G. Licciardi, R. Restaino, and L. Wald, “A Critical Comparison Among Pansharpening Algorithms”, 
%                           IEEE Transaction on Geoscience and Remote Sensing, 2014. (Accepted)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function I_Interpolated = upsampling7tap(I_Interpolated,ratio)

if (2^round(log2(ratio)) ~= ratio)
    disp('Error: Only resize factors power of 2');
    return;
end 

[r,c,b] = size(I_Interpolated);
CDF7 = [-2*0.045635881557 -2*0.028771763114 2*0.295635881557 2*0.557543526229 2*0.295635881557 -2*0.028771763114 -2*0.045635881557];
BaseCoeff = CDF7;
first = 1;

for z = 1 : ratio/2

    I1LRU = zeros((2^z) * r, (2^z) * c, b);
    if first
        I1LRU(2:2:end,2:2:end,:) = I_Interpolated;
        first = 0;
    else
        I1LRU(1:2:end,1:2:end,:) = I_Interpolated;
    end

    for ii = 1 : b
        t = I1LRU(:,:,ii); 
        t = imfilter(t',BaseCoeff,'circular'); 
        I1LRU(:,:,ii) = imfilter(t',BaseCoeff,'circular'); 
    end

    I_Interpolated = I1LRU;
    
end

end