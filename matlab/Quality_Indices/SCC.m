
function sCC = SCC(I_F,I_GT)
if iscell(I_GT)
    I_GT=cell2mat(I_GT);
end
if iscell(I_F)
    I_F=cell2mat(I_F);
end
%%% sCC
Im_Lap_F = zeros(size(I_F));
for idim=1:size(I_F,3),
    Im_Lap_F(:,:,idim)= imfilter(I_F(:,:,idim),fspecial('sobel'));
end
Im_Lap_GT = zeros(size(I_GT));
for idim=1:size(I_GT,3),
    Im_Lap_GT(:,:,idim)= imfilter(I_GT(:,:,idim),fspecial('sobel'));
end


sCC = sum(sum(sum(Im_Lap_GT.*Im_Lap_F)));
sCC = sCC/sqrt(sum(sum(sum((Im_Lap_GT.^2)))));
sCC = sCC/sqrt(sum(sum(sum((Im_Lap_F.^2)))));

end