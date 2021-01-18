%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           GS2_GLP fuses the upsampled MultiSpectral (MS) and PANchromatic (PAN) images by 
%           exploiting the Gram-Schmidt (GS) mode 2 algorithm with Generalized Laplacian Pyramid (GLP) decomposition.
% 
% Interface:
%           I_Fus_GS2_GLP = GS2_GLP(I_MS,I_PAN,ratio,sensor,tag)
%
% Inputs:
%           I_MS:           MS image upsampled at PAN scale;
%           I_PAN:          PAN image;
%           ratio:          Scale ratio between MS and PAN. Pre-condition: Integer value;
%           sensor:         String for type of sensor (e.g. 'WV2','IKONOS');
%           tag:            Image tag. Often equal to the field sensor. It makes sense when sensor is 'none'. It indicates the band number
%                           in the latter case.
%
% Outputs:
%           I_Fus_GS2_GLP:  GS2_GLP pasharpened image.
% 
% References:
%           [Aiazzi06]      B. Aiazzi, L. Alparone, S. Baronti, A. Garzelli, and M. Selva, “MTF-tailored multiscale fusion of high-resolution MS and Pan imagery,?
%                           Photogrammetric Engineering and Remote Sensing, vol. 72, no. 5, pp. 591?96, May 2006.
%           [Alparone07]    L. Alparone, L. Wald, J. Chanussot, C. Thomas, P. Gamba, and L. M. Bruce, “Comparison of pansharpening algorithms: Outcome
%                           of the 2006 GRS-S Data Fusion Contest,?IEEE Transactions on Geoscience and Remote Sensing, vol. 45, no. 10, pp. 3012?021,
%                           October 2007.
%           [Vivone14]      G. Vivone, L. Alparone, J. Chanussot, M. Dalla Mura, A. Garzelli, G. Licciardi, R. Restaino, and L. Wald, “A Critical Comparison Among Pansharpening Algorithms? 
%                           IEEE Transaction on Geoscience and Remote Sensing, 2014. (Accepted)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function I_Fus_GS2_GLP = GS2_GLP(I_MS,I_PAN,ratio,sensor,tag)

imageLR = double(I_MS);
imageHR = double(I_PAN);

imageHR = repmat(imageHR,[1 1 size(imageLR,3)]);

switch sensor
    case 'QB' 
        GNyq = [0.34 0.32 0.30 0.22]; % Band Order: B,G,R,NIR
    case 'IKONOS'
        GNyq = [0.26,0.28,0.29,0.28]; % Band Order: B,G,R,NIR
    case 'GeoEye1'
        GNyq = [0.23,0.23,0.23,0.23]; % Band Order: B,G,R,NIR
    case 'WV2'
        GNyq = [0.35 .* ones(1,7), 0.27];%GNyq = [0.35, 0.35,0.35,0.35];% 
    case 'WV3'
        GNyq = [ 0.355, 0.360,0.365,0.335];%GNyq = [0.325, 0.355, 0.360, 0.350, 0.365, 0.360, 0.335, 0.315];
    case 'none'
        if strcmp(tag,'WV2')
            GNyq = 0.15 .* ones(1,8);
        else
            GNyq = 0.29 .* ones(1,size(I_MS,3)); 
        end
end


%%% MTF
N = 41;
PAN_LP = zeros(size(I_MS));
nBands = size(I_MS,3);
fcut = 1/ratio;

for ii = 1 : nBands
    alpha = sqrt((N*(fcut/2))^2/(-2*log(GNyq(ii))));
    H = fspecial('gaussian', N, alpha);
    Hd = H./max(H(:));
    h = fwind1(Hd,kaiser(N));
    PAN_LP(:,:,ii) = imfilter(imageHR(:,:,ii),real(h),'replicate');
    t = imresize(PAN_LP(:,:,ii),1/ratio,'nearest');
    PAN_LP(:,:,ii) = interp23tap(t,ratio);
end

PAN_LP = double(PAN_LP);

%%% Coefficients
g = ones(1,size(I_MS,3));
for ii = 1 : size(I_MS,3)
    h = imageLR(:,:,ii);
    h2 = PAN_LP(:,:,ii);
    c = cov(h2(:),h(:));
    g(ii) = c(1,2)/var(h2(:));
end

%%% Detail Extraction
delta = imageHR - PAN_LP;

I_Fus_GS2_GLP = zeros(size(imageLR));

for ii = 1 : size(imageLR,3)
    I_Fus_GS2_GLP(:,:,ii) = imageLR(:,:,ii) + delta(:,:,ii) .* g(ii);
end

end