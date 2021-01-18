%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           MTF filters the image I_PAN using a Gaussin filter matched with the Modulation Transfer Function (MTF) of the PANchromatic (PAN) sensor. 
% 
% Interface:
%           I_Filtered = MTF_PAN(I_PAN,sensor,ratio)
%
% Inputs:
%           I_PAN:          PAN image;
%           sensor:         String for type of sensor (e.g. 'WV2', 'IKONOS');
%           ratio:          Scale ratio between MS and PAN.
%
% Outputs:
%           I_Filtered:     Output filtered PAN image.
% 
% References:
%           [Aiazzi06]      B. Aiazzi, L. Alparone, S. Baronti, A. Garzelli, and M. Selva, “MTF-tailored multiscale fusion of high-resolution MS and Pan imagery,?
%                           Photogrammetric Engineering and Remote Sensing, vol. 72, no. 5, pp. 591?96, May 2006.
%           [Lee10]         J. Lee and C. Lee, “Fast and efficient panchromatic sharpening,?IEEE Transactions on Geoscience and Remote Sensing, vol. 48, no. 1,
%                           pp. 155?63, January 2010.
%           [Vivone14]      G. Vivone, L. Alparone, J. Chanussot, M. Dalla Mura, A. Garzelli, G. Licciardi, R. Restaino, and L. Wald, “A Critical Comparison Among Pansharpening Algorithms? 
%                           IEEE Transaction on Geoscience and Remote Sensing, 2014. (Accepted)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function I_Filtered = MTF_PAN(I_PAN,sensor,ratio)

switch sensor
    case 'QB' 
        GNyq = 0.15; 
    case 'IKONOS'
        GNyq = 0.17;
    case 'GeoEye1'
        GNyq = 0.16;
    case 'WV2'
        GNyq = 0.11;
    case 'WV3'
        GNyq = 0.5;  
    case 'none'
        GNyq = 0.15;
end


N = 41;
fcut = 1/ratio;
 
alpha = sqrt((N*(fcut/2))^2/(-2*log(GNyq)));
H = fspecial('gaussian', N, alpha);
Hd = H./max(H(:));
h = fwind1(Hd,kaiser(N));
I_PAN_LP = imfilter(I_PAN,real(h),'replicate');

I_Filtered= double(I_PAN_LP);

end