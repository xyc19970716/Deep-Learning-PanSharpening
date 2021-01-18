%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           HPF fuses the upsampled MultiSpectral (MS) and PANchromatic (PAN) images by 
%           exploiting the High Pass Filtering (HPF) pansharpening algorithm. 
% 
% Interface:
%           I_Fus_HPF = HPF(I_MS,I_PAN,ratio)
%
% Inputs:
%           I_MS:               MS image upsampled at PAN scale;
%           I_PAN:              PAN image;
%           ratio:              Scale ratio between MS and PAN. Pre-condition: Integer value.
%
% Outputs:
%           I_Fus_HPF:          HPF pasharpened image.
% 
% References:
%           [Chavez91]          P. S. Chavez Jr., S. C. Sides, and J. A. Anderson, “Comparison of three different methods to merge multiresolution and multispectral
%                               data: Landsat TM and SPOT panchromatic,” Photogrammetric Engineering and Remote Sensing, vol. 57, no. 3, pp. 295–303, March 1991.
%           [Schowengerdt97]    R. A. Schowengerdt, Remote Sensing: Models and Methods for Image Processing, 2nd ed. Orlando, FL: Academic, 1997.
%           [Vivone14]          G. Vivone, L. Alparone, J. Chanussot, M. Dalla Mura, A. Garzelli, G. Licciardi, R. Restaino, and L. Wald, “A Critical Comparison Among Pansharpening Algorithms”, 
%                               IEEE Transaction on Geoscience and Remote Sensing, 2014. (Accepted)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function I_Fus_HPF = HPF(I_MS,I_PAN,ratio)

if ~ rem(ratio,2)
    ratio = ratio + 1;
end

I_PAN = repmat(I_PAN,[1 1 size(I_MS,3)]);

for ii = 1 : size(I_MS,3)    
  I_PAN(:,:,ii) = (I_PAN(:,:,ii) - mean2(I_PAN(:,:,ii))).*(std2(I_MS(:,:,ii))./std2(I_PAN(:,:,ii))) + mean2(I_MS(:,:,ii));  
end

[Height,Width,Bands]=size(I_MS);
I_Fus_HPF=zeros(Height,Width,Bands,'double');


for i=1:Bands
    h = fspecial('average',[ratio ratio]);
    P_LP = imfilter(I_PAN(:,:,i),h,'replicate');
    PanDetails= I_PAN(:,:,i) - P_LP;
    I_Fus_HPF(:,:,i)=I_MS(:,:,i)+PanDetails;
end

end

