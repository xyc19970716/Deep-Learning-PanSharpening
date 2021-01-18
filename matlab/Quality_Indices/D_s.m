%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           Quality with No Reference (QNR). Spatial distortion index.
% 
% Interface:
%           D_s_index = D_s(I_F,I_MS,I_PAN,sensor,ratio,S,q)
%
% Inputs:
%           I_F:                Pansharpened image;
%           I_MS:               MS image resampled to panchromatic scale;
%           I_PAN:              Panchromatic image;
%           sensor:             String for type of sensor (e.g. 'WV2','IKONOS');
%           ratio:              Scale ratio between MS and PAN. Pre-condition: Integer value;
%           S:                  Block size (optional); Default value: 32;
%           q:                  Exponent value (optional); Default value: q = 1.
% 
% Outputs:
%           D_s_index:          D_s index.
% 
% References:
%           [Alparone08]        L. Alparone, B. Aiazzi, S. Baronti, A. Garzelli, F. Nencini, and M. Selva, "Multispectral and panchromatic data fusion assessment without reference,"
%                               Photogrammetric Engineering and Remote Sensing, vol. 74, no. 2, pp. 193?00, February 2008. 
%           [Vivone14]          G. Vivone, L. Alparone, J. Chanussot, M. Dalla Mura, A. Garzelli, G. Licciardi, R. Restaino, and L. Wald, “A Critical Comparison Among Pansharpening Algorithms? 
%                               IEEE Transaction on Geoscience and Remote Sensing, 2014. (Accepted)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function D_s_index = D_s(I_F,I_MS,I_PAN,sensor,ratio,S,q)
if iscell(I_F)
    I_F=cell2mat(I_F);
end
if iscell(I_MS)
    I_MS=cell2mat(I_MS);
end
if iscell(I_PAN)
    I_PAN=cell2mat(I_PAN);
end
if (size(I_F) ~= size(I_MS))
    error('The two images must have the same dimensions')
end

[N, M, Nb] = size(I_F);

if (rem(N,S) ~= 0)
    error('number of rows must be multiple of the block size')
end

if (rem(M,S) ~= 0)
    error('number of columns must be multiple of the block size')
end

pan_filt = MTF_PAN(I_PAN,sensor,ratio);

D_s_index = 0;
for i = 1:Nb
        band1 = I_F(:,:,i);
        band2 = I_PAN;
        fun_uqi = @(bs) uqi(bs.data,...
            band2(bs.location(1):bs.location(1)+S-1,...
            bs.location(2):bs.location(2)+S-1));
        Qmap_high = blockproc(band1,[S S],fun_uqi);
        Q_high = mean2(Qmap_high);
        band1 = I_MS(:,:,i);
        band2 = pan_filt;
        fun_uqi = @(bs) uqi(bs.data,...
            band2(bs.location(1):bs.location(1)+S-1,...
            bs.location(2):bs.location(2)+S-1));
        Qmap_low = blockproc(band1,[S S],fun_uqi);
        Q_low = mean2(Qmap_low);
        D_s_index = D_s_index + abs(Q_high-Q_low)^q;
end

D_s_index = (D_s_index/Nb)^(1/q);

end

%%%%%%% Q-index on x and y images
function Q = uqi(x,y)

x = double(x(:));
y = double(y(:));
mx = mean(x);
my = mean(y);
C = cov(x,y);

Q = 4 * C(1,2) * mx * my / (C(1,1)+C(2,2)) / (mx^2 + my^2);  

end