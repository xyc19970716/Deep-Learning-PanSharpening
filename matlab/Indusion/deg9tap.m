%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           deg9tap filters and decimates the image I_Filtered using a polynomial with 9 coefficients filter. 
% 
% Interface:
%           I_Filtered = deg9tap(I_Filtered,ratio)
%
% Inputs:
%           I_Filtered:     Image to be filtered and decimated;
%           ratio:          Scale ratio between MS and PAN. Pre-condition: Resize factors power of 2.
%
% Outputs:
%           I_Filtered:     Filtered and decimated image.
% 
% References:
%           [Khan08]        M. M. Khan, J. Chanussot, L. Condat, and A. Montavert, “Indusion: Fusion of multispectral and panchromatic images using the
%                           induction scaling technique,” IEEE Geoscience and Remote Sensing Letters, vol. 5, no. 1, pp. 98–102, January 2008.
%           [Aiazzi02]      B. Aiazzi, L. Alparone, S. Baronti, and A. Garzelli, “Context-driven fusion of high spatial and spectral resolution images based on
%                           oversampled multiresolution analysis,” IEEE Transactions on Geoscience and Remote Sensing, vol. 40, no. 10, pp. 2300–2312, October
%                           2002.
%           [Vivone14]      G. Vivone, L. Alparone, J. Chanussot, M. Dalla Mura, A. Garzelli, G. Licciardi, R. Restaino, and L. Wald, “A Critical Comparison Among Pansharpening Algorithms”, 
%                           IEEE Transaction on Geoscience and Remote Sensing, 2014. (Accepted)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function I_Filtered = deg9tap(I_Filtered,ratio)

if (2^round(log2(ratio)) ~= ratio)
    disp('Error: Only resize factors power of 2');
    return;
end 

[~,~,b] = size(I_Filtered);

CDF9=[0.026748757411 -0.016864118443 -0.078223266529 0.266864118443 0.602949018236 0.266864118443 -0.078223266529 -0.016864118443 0.026748757411];
BaseCoeff = CDF9;
first = 1;

for z = 1 : ratio/2
    
    h = zeros(size(I_Filtered));
    for ii = 1 : b
        t = I_Filtered(:,:,ii); 
        t = imfilter(t',BaseCoeff,'replicate'); 
        h(:,:,ii) = imfilter(t',BaseCoeff,'replicate'); 
    end
    
    if first
        h = h(2:2:end,2:2:end,:);
        first = 0;
    else
        h = h(1:2:end,1:2:end,:);
    end

    I_Filtered = h;
end

end