%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           AWLP fuses the upsampled MultiSpectral (MS) and PANchromatic (PAN) images by 
%           exploiting the Additive Wavelet Luminance Proportional (AWLP) algorithm.
% 
% Interface:
%           I_Fus_AWLP = AWLP(I_MS,I_PAN,ratio)
%
% Inputs:
%           I_MS:           MS image upsampled at PAN scale;
%           I_PAN:          PAN image;
%           ratio:          Scale ratio between MS and PAN. Pre-condition: Integer value.
%
% Outputs:
%           I_Fus_AWLP:     AWLP pasharpened image.
% 
% References:
%           [Otazu05]       X. Otazu, M. Gonz´alez-Aud´?cana, O. Fors, and J. N´u˜nez, “Introduction of sensor spectral response into image fusion methods.
%                           Application to wavelet-based methods,” IEEE Transactions on Geoscience and Remote Sensing, vol. 43, no. 10, pp. 2376–2385,
%                           October 2005.
%           [Alparone07]    L. Alparone, L. Wald, J. Chanussot, C. Thomas, P. Gamba, and L. M. Bruce, “Comparison of pansharpening algorithms: Outcome
%                           of the 2006 GRS-S Data Fusion Contest,” IEEE Transactions on Geoscience and Remote Sensing, vol. 45, no. 10, pp. 3012–3021,
%                           October 2007.
%           [Vivone14]      G. Vivone, L. Alparone, J. Chanussot, M. Dalla Mura, A. Garzelli, G. Licciardi, R. Restaino, and L. Wald, “A Critical Comparison Among Pansharpening Algorithms”, 
%                           IEEE Transaction on Geoscience and Remote Sensing, 2014. (Accepted)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function I_Fus_AWLP = AWLP(I_MS,I_PAN,ratio)

[Height,Width,Bands]=size(I_MS);
I_Fus_AWLP=zeros(Height,Width,Bands,'double');

SumImage=sum(I_MS,3)/Bands;

IntensityRatio = zeros(size(I_MS));
for i=1:Bands
    IntensityRatio(:,:,i)=I_MS(:,:,i)./(SumImage+eps);
end

I_PAN = repmat(I_PAN,[1 1 size(I_MS,3)]);

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

%%%%%%%%% OLD [as in the article Otazu05]
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

    I_Fus_AWLP(:,:,i) = StepDetails .* IntensityRatio(:,:,i)+I_MS(:,:,i);
end

end