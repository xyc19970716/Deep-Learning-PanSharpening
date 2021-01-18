%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           Indusion fuses the MultiSpectral (MS) and PANchromatic (PAN) images by 
%           exploiting the Indusion algorithm. 
% 
% Interface:
%           I_Fus_Indusion = Indusion(I_PAN,I_MS_LR,ratio)
%
% Inputs:
%           I_PAN:          PAN image;
%           I_MS_LR:        MS image;
%           ratio:          Scale ratio between MS and PAN. Pre-condition: Integer value.
%
% Outputs:
%           I_Fus_Indusion: Indusion pansharpened image.
% 
% References:
%           [Khan08]        M. M. Khan, J. Chanussot, L. Condat, and A. Montavert, “Indusion: Fusion of multispectral and panchromatic images using the
%                           induction scaling technique,” IEEE Geoscience and Remote Sensing Letters, vol. 5, no. 1, pp. 98–102, January 2008.
%           [Vivone14]      G. Vivone, L. Alparone, J. Chanussot, M. Dalla Mura, A. Garzelli, G. Licciardi, R. Restaino, and L. Wald, “A Critical Comparison Among Pansharpening Algorithms”, 
%                           IEEE Transaction on Geoscience and Remote Sensing, 2014. (Accepted)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function I_Fus_Indusion = Indusion(I_PAN,I_MS_LR,ratio)

    [High_Res_Height,High_Res_Width]=size(I_PAN);
    Number_Of_Bands = size(I_MS_LR,3);
    I_Fus_Indusion = zeros(High_Res_Height,High_Res_Width,Number_Of_Bands);
    Low_Resolution_Pan = deg9tap(I_PAN,ratio);
    
    for i=1:Number_Of_Bands
        LowRes_MS_Band = I_MS_LR(:,:,i);        
        Low_Resolution_Pan_Eq = (Low_Resolution_Pan - mean2(Low_Resolution_Pan)).*(std2(LowRes_MS_Band)./std2(Low_Resolution_Pan)) + mean2(LowRes_MS_Band);
        MS_Band = upsampling7tap(LowRes_MS_Band,ratio);       
        Low_Resolution_Pan_Eq = upsampling7tap(Low_Resolution_Pan_Eq,ratio);
        High_Resolution_Pan_Eq = (I_PAN - mean2(I_PAN)).*(std2(MS_Band)./std2(I_PAN)) + mean2(MS_Band);
        DifferenceImage = High_Resolution_Pan_Eq - Low_Resolution_Pan_Eq;
        I_Fus_Indusion(:,:,i) = MS_Band + DifferenceImage;
    end
    
end