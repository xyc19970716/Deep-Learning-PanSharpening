%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           ATWT fuses the upsampled MultiSpectral (MS) and PANchromatic (PAN) images by 
%           exploiting the A Trous Wavelet Transform (ATWT) and additive injection model algorithm.
% 
% Interface:
%           I_Fus_ATWT = ATWT(I_MS,I_PAN,ratio)
%
% Inputs:
%           I_MS:           MS image upsampled at PAN scale;
%           I_PAN:          PAN image;
%           ratio:          Scale ratio between MS and PAN. Pre-condition: Integer value.
%
% Outputs:
%           I_Fus_ATWT:     ATWT pasharpened image.
% 
% References:
%           [Nunez99]       J. Nunez, X. Otazu, O. Fors, A. Prades, V. Pala, and R. Arbiol, “Multiresolution-based image fusion with additive wavelet
%                           decomposition,” IEEE Transactions on Geoscience and Remote Sensing, vol. 37, no. 3, pp. 1204–1211, May 1999.
%           [Vivone14a]     G. Vivone, R. Restaino, M. Dalla Mura, G. Licciardi, and J. Chanussot, “Contrast and error-based fusion schemes for multispectral
%                           image pansharpening,” IEEE Geoscience and Remote Sensing Letters, vol. 11, no. 5, pp. 930–934, May 2014.
%           [Vivone14b]     G. Vivone, L. Alparone, J. Chanussot, M. Dalla Mura, A. Garzelli, G. Licciardi, R. Restaino, and L. Wald, “A Critical Comparison Among Pansharpening Algorithms”, 
%                           IEEE Transaction on Geoscience and Remote Sensing, 2014. (Accepted)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function I_Fus_ATWT = ATWT(I_MS,I_PAN,ratio)

[Height,Width,Bands]=size(I_MS);
I_Fus_ATWT=zeros(Height,Width,Bands,'double');

I_PAN = repmat(I_PAN,[1 1 size(I_MS,3)]);

%%% Different w.r.t. [Nunez99] in which the equalization is done
%%% w.r.t. the Luminance of the MS image

%%% Equalization for each band 
for ii = 1 : size(I_MS,3)    
  I_PAN(:,:,ii) = (I_PAN(:,:,ii) - mean2(I_PAN(:,:,ii))).*(std2(I_MS(:,:,ii))./std2(I_PAN(:,:,ii))) + mean2(I_MS(:,:,ii));  
end

h=[1 4 6 4 1 ]/16;
g=[0 0 1 0 0 ]-h;
htilde=[ 1 4 6 4 1]/16;
gtilde=[ 0 0 1 0 0 ]+htilde;
h=sqrt(2)*h;
g=sqrt(2)*g;
htilde=sqrt(2)*htilde;
gtilde=sqrt(2)*gtilde;
WF={h,g,htilde,gtilde};

Levels = ceil(log2(ratio));

for i=1:Bands    
    WT = ndwt2(I_PAN(:,:,i),Levels,WF);
    
    for ii = 2 : numel(WT.dec), WT.dec{ii} = zeros(size(WT.dec{ii})); end
    
    StepDetails = I_PAN(:,:,i) - indwt2(WT,'c');
    
%%%%%%%%% OLD (as in the article [Nunez99]). Lower performances.
%     sINI = WT.sizeINI;
%     
%     StepDetails = zeros(sINI);
%     
%     for ii = 2 : numel(WT.dec)
%         h = WT.dec{ii};
%         h = imcrop(h,[(size(h,1) - sINI(1))/2 + 1,(size(h,2) - sINI(2))/2 + 1, sINI(1) - 1, sINI(2) - 1]);
%         StepDetails = StepDetails + h; 
%     end
%%%%%%%%%%%%%%%%%%%
    
    I_Fus_ATWT(:,:,i) = StepDetails + I_MS(:,:,i);
end

end