%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           Q/SSIM averaged on all bands.
% 
% Interface:
%           Q_avg = Q(I1,I2,L)
%
% Inputs:
%           I1:         First multispectral image;
%           I2:         Second multispectral image;
%           L:          Radiometric resolution.
%
% Outputs:
%           Q_avg:      Q index averaged on all bands.
% 
% References:
%           [Wang02]    Z. Wang and A. C. Bovik, “A universal image quality index,?IEEE Signal Processing Letters, vol. 9, no. 3, pp. 81?4, March 2002.
%           [Vivone14]  G. Vivone, L. Alparone, J. Chanussot, M. Dalla Mura, A. Garzelli, G. Licciardi, R. Restaino, and L. Wald, “A Critical Comparison Among Pansharpening Algorithms? 
%                       IEEE Transaction on Geoscience and Remote Sensing, 2014. (Accepted)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Q_avg = Q(I1,I2,L)
if iscell(I1)
    I1=cell2mat(I1);
end
if iscell(I2)
    I2=cell2mat(I2);
end

Q_orig = zeros(1,size(I1,3));

for idim=1:size(I1,3),
%     Q_orig(idim) = ssim(I_GT(:,:,idim),I1U(:,:,idim), [0.01 0.03],fspecial('gaussian', 11, 1.5), L);
    Q_orig(idim) = img_qi(I1(:,:,idim),I2(:,:,idim), 32);
end

Q_avg = mean(Q_orig);

end