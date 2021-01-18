%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%  RUN AND REDUCED RESOLUTION VALIDATION  %%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Initialization of the Matrix of Results
NumAlgs = 19;
NumIndexes = 5;
MatrixResults = zeros(NumAlgs,NumIndexes);

%% MS

if size(I_GT,3) == 4   
    showImage4LR(I_MS_LR,printEPS,3,flag_cut_bounds,dim_cut,thvalues,L,ratio);    
else
    showImage8LR(I_MS_LR,printEPS,3,flag_cut_bounds,dim_cut,thvalues,L,ratio);
end

%% PAN

showPan(I_PAN,printEPS,2,flag_cut_bounds,dim_cut);

%% GT

[Q_avg_GT, SAM_GT, ERGAS_GT, SCC_GT_GT, Q_GT] = indexes_evaluation(I_GT,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);

if size(I_GT,3) == 4    
    showImage4(I_GT,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L);
else
    showImage8(I_GT,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L);
end

%% EXP

[Q_avg_EXP, SAM_EXP, ERGAS_EXP, SCC_GT_EXP, Q_EXP] = indexes_evaluation(I_MS,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);

if size(I_GT,3) == 4    
    showImage4(I_MS,printEPS,3,flag_cut_bounds,dim_cut,thvalues,L);    
else
    showImage8(I_MS,printEPS,3,flag_cut_bounds,dim_cut,thvalues,L);
end

MatrixResults(1,:) = [Q_EXP,Q_avg_EXP,SAM_EXP,ERGAS_EXP,SCC_GT_EXP];

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Component Substitution %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% PCA

cd PCA
t2=tic;
I_PCA = PCA(I_MS,I_PAN);
time_PCA=toc(t2);
fprintf('Elaboration time PCA: %.2f [sec]\n',time_PCA);
cd ..

[Q_avg_PCA, SAM_PCA, ERGAS_PCA, SCC_GT_PCA, Q_PCA] = indexes_evaluation(I_PCA,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);

if size(I_GT,3) == 4    
    showImage4(I_PCA,printEPS,4,flag_cut_bounds,dim_cut,thvalues,L);
else
    showImage8(I_PCA,printEPS,4,flag_cut_bounds,dim_cut,thvalues,L);
end

MatrixResults(2,:) = [Q_PCA,Q_avg_PCA,SAM_PCA,ERGAS_PCA,SCC_GT_PCA];

%% IHS

cd IHS
t2=tic;
I_IHS = IHS(I_MS,I_PAN);
time_IHS=toc(t2);
fprintf('Elaboration time IHS: %.2f [sec]\n',time_IHS);
cd ..

[Q_avg_IHS, SAM_IHS, ERGAS_IHS, SCC_GT_IHS, Q_IHS] = indexes_evaluation(I_IHS,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);

if size(I_GT,3) == 4
    showImage4(I_IHS,printEPS,5,flag_cut_bounds,dim_cut,thvalues,L);
else
    showImage8(I_IHS,printEPS,5,flag_cut_bounds,dim_cut,thvalues,L);
end

MatrixResults(3,:) = [Q_IHS,Q_avg_IHS,SAM_IHS,ERGAS_IHS,SCC_GT_IHS];

%% Brovey

cd Brovey
t2=tic;
I_Brovey = Brovey(I_MS,I_PAN);
time_Brovey=toc(t2);
fprintf('Elaboration time Brovey: %.2f [sec]\n',time_Brovey);
cd ..

[Q_avg_Brovey, SAM_Brovey, ERGAS_Brovey, SCC_GT_Brovey, Q_Brovey] = indexes_evaluation(I_Brovey,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);

if size(I_GT,3) == 4
    showImage4(I_Brovey,printEPS,6,flag_cut_bounds,dim_cut,thvalues,L);
else
    showImage8(I_Brovey,printEPS,6,flag_cut_bounds,dim_cut,thvalues,L);
end

MatrixResults(4,:) = [Q_Brovey,Q_avg_Brovey,SAM_Brovey,ERGAS_Brovey,SCC_GT_Brovey];

%% BDSD

cd BDSD
t2=tic;

I_BDSD = BDSD(I_MS,I_PAN,ratio,size(I_MS,1),sensor);

time_BDSD = toc(t2);
fprintf('Elaboration time BDSD: %.2f [sec]\n',time_BDSD);
cd ..

[Q_avg_BDSD, SAM_BDSD, ERGAS_BDSD, SCC_GT_BDSD, Q_BDSD] = indexes_evaluation(I_BDSD,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);

if size(I_GT,3) == 4
    showImage4(I_BDSD,printEPS,7,flag_cut_bounds,dim_cut,thvalues,L);    
else
    showImage8(I_BDSD,printEPS,7,flag_cut_bounds,dim_cut,thvalues,L);
end

MatrixResults(5,:) = [Q_BDSD,Q_avg_BDSD,SAM_BDSD,ERGAS_BDSD,SCC_GT_BDSD];

%% GS

cd GS
t2=tic;
I_GS = GS(I_MS,I_PAN);
time_GS = toc(t2);
fprintf('Elaboration time GS: %.2f [sec]\n',time_GS);
cd ..

[Q_avg_GS, SAM_GS, ERGAS_GS, SCC_GT_GS, Q_GS] = indexes_evaluation(I_GS,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);

if size(I_GT,3) == 4   
    showImage4(I_GS,printEPS,8,flag_cut_bounds,dim_cut,thvalues,L);
else
    showImage8(I_GS,printEPS,8,flag_cut_bounds,dim_cut,thvalues,L);
end

MatrixResults(6,:) = [Q_GS,Q_avg_GS,SAM_GS,ERGAS_GS,SCC_GT_GS];

%% GSA

cd GS
t2=tic;
I_GSA = GSA(I_MS,I_PAN,I_MS_LR,ratio);
tempo_GSA = toc(t2);
fprintf('Elaboration time GSA: %.2f [sec]\n',tempo_GSA);
cd ..

[Q_avg_GSA, SAM_GSA, ERGAS_GSA, SCC_GT_GSA, Q_GSA] = indexes_evaluation(I_GSA,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);

if size(I_GT,3) == 4
    showImage4(I_GSA,printEPS,9,flag_cut_bounds,dim_cut,thvalues,L);
else
    showImage8(I_GSA,printEPS,9,flag_cut_bounds,dim_cut,thvalues,L);
end

MatrixResults(7,:) = [Q_GSA,Q_avg_GSA,SAM_GSA,ERGAS_GSA,SCC_GT_GSA];

%% PRACS

cd PRACS
t2=tic;
I_PRACS = PRACS(I_MS,I_PAN,ratio);
time_PRACS = toc(t2);
fprintf('Elaboration time PRACS: %.2f [sec]\n',time_PRACS);
cd ..

[Q_avg_PRACS, SAM_PRACS, ERGAS_PRACS, SCC_GT_PRACS, Q_PRACS] = indexes_evaluation(I_PRACS,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);

if size(I_GT,3) == 4
    showImage4(I_PRACS,printEPS,10,flag_cut_bounds,dim_cut,thvalues,L);
else
    showImage8(I_PRACS,printEPS,10,flag_cut_bounds,dim_cut,thvalues,L);
end

MatrixResults(8,:) = [Q_PRACS,Q_avg_PRACS,SAM_PRACS,ERGAS_PRACS,SCC_GT_PRACS];


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% MultiResolution Analysis %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% HPF

cd HPF
t2=tic;
I_HPF = HPF(I_MS,I_PAN,ratio);
time_HPF = toc(t2);
fprintf('Elaboration time HPF: %.2f [sec]\n',time_HPF);
cd ..

[Q_avg_HPF, SAM_HPF, ERGAS_HPF, SCC_GT_HPF, Q_HPF] = indexes_evaluation(I_HPF,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);

if size(I_GT,3) == 4
    showImage4(I_HPF,printEPS,11,flag_cut_bounds,dim_cut,thvalues,L);
else
    showImage8(I_HPF,printEPS,11,flag_cut_bounds,dim_cut,thvalues,L);
end

MatrixResults(9,:) = [Q_HPF,Q_avg_HPF,SAM_HPF,ERGAS_HPF,SCC_GT_HPF];

%% SFIM

cd SFIM
t2=tic;
I_SFIM = SFIM(I_MS,I_PAN,ratio);
time_SFIM = toc(t2);
fprintf('Elaboration time SFIM: %.2f [sec]\n',time_SFIM);
cd ..

[Q_avg_SFIM, SAM_SFIM, ERGAS_SFIM, SCC_GT_SFIM, Q_SFIM] = indexes_evaluation(I_SFIM,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);

if size(I_GT,3) == 4
    showImage4(I_SFIM,printEPS,12,flag_cut_bounds,dim_cut,thvalues,L);
else
    showImage8(I_SFIM,printEPS,12,flag_cut_bounds,dim_cut,thvalues,L);
end

MatrixResults(10,:) = [Q_SFIM,Q_avg_SFIM,SAM_SFIM,ERGAS_SFIM,SCC_GT_SFIM];

%% Indusion

cd Indusion
t2=tic;
I_Indusion = Indusion(I_PAN,I_MS_LR,ratio);
time_Indusion = toc(t2);
fprintf('Elaboration time Indusion: %.2f [sec]\n',time_Indusion);
cd ..

[Q_avg_Indusion, SAM_Indusion, ERGAS_Indusion, SCC_GT_Indusion, Q_Indusion] = indexes_evaluation(I_Indusion,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);

if size(I_GT,3) == 4
    showImage4(I_Indusion,printEPS,13,flag_cut_bounds,dim_cut,thvalues,L);
else
    showImage8(I_Indusion,printEPS,13,flag_cut_bounds,dim_cut,thvalues,L);
end

MatrixResults(11,:) = [Q_Indusion,Q_avg_Indusion,SAM_Indusion,ERGAS_Indusion,SCC_GT_Indusion];

%% ATWT

cd Wavelet
t2=tic;
I_ATWT = ATWT(I_MS,I_PAN,ratio);
time_ATWT = toc(t2);
fprintf('Elaboration time ATWT: %.2f [sec]\n',time_ATWT);
cd ..

[Q_avg_ATWT, SAM_ATWT, ERGAS_ATWT, SCC_GT_ATWT, Q_ATWT] = indexes_evaluation(I_ATWT,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);

if size(I_GT,3) == 4
    showImage4(I_ATWT,printEPS,14,flag_cut_bounds,dim_cut,thvalues,L);
else
    showImage8(I_ATWT,printEPS,14,flag_cut_bounds,dim_cut,thvalues,L);
end

MatrixResults(12,:) = [Q_ATWT,Q_avg_ATWT,SAM_ATWT,ERGAS_ATWT,SCC_GT_ATWT];

%% AWLP

cd Wavelet
t2=tic;
I_AWLP = AWLP(I_MS,I_PAN,ratio);
time_AWLP = toc(t2);
fprintf('Elaboration time AWLP: %.2f [sec]\n',time_AWLP);
cd ..

[Q_avg_AWLP, SAM_AWLP, ERGAS_AWLP, SCC_GT_AWLP, Q_AWLP] = indexes_evaluation(I_AWLP,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);

if size(I_GT,3) == 4
    showImage4(I_AWLP,printEPS,15,flag_cut_bounds,dim_cut,thvalues,L);
else
    showImage8(I_AWLP,printEPS,15,flag_cut_bounds,dim_cut,thvalues,L);
end

MatrixResults(13,:) = [Q_AWLP,Q_avg_AWLP,SAM_AWLP,ERGAS_AWLP,SCC_GT_AWLP];

%% ATWT-M2

cd Wavelet
t2=tic;

I_ATWTM2 = ATWT_M2(I_MS,I_PAN,ratio);

time_ATWTM2 = toc(t2);
fprintf('Elaboration time ATWT-M2: %.2f [sec]\n',time_ATWTM2);
cd ..

[Q_avg_ATWTM2, SAM_ATWTM2, ERGAS_ATWTM2, SCC_GT_ATWTM2, Q_ATWTM2] = indexes_evaluation(I_ATWTM2,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);

if size(I_GT,3) == 4   
    showImage4(I_ATWTM2,printEPS,16,flag_cut_bounds,dim_cut,thvalues,L);    
else
    showImage8(I_ATWTM2,printEPS,16,flag_cut_bounds,dim_cut,thvalues,L);
end

MatrixResults(14,:) = [Q_ATWTM2,Q_avg_ATWTM2,SAM_ATWTM2,ERGAS_ATWTM2,SCC_GT_ATWTM2];

%% ATWT-M3

cd Wavelet
t2=tic;

I_ATWTM3 = ATWT_M3(I_MS,I_PAN,ratio);

time_ATWTM3 = toc(t2);
fprintf('Elaboration time ATWT-M3: %.2f [sec]\n',time_ATWTM3);
cd ..

[Q_avg_ATWTM3, SAM_ATWTM3, ERGAS_ATWTM3, SCC_GT_ATWTM3, Q_ATWTM3] = indexes_evaluation(I_ATWTM3,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);

if size(I_GT,3) == 4   
    showImage4(I_ATWTM3,printEPS,17,flag_cut_bounds,dim_cut,thvalues,L);
else
    showImage8(I_ATWTM3,printEPS,17,flag_cut_bounds,dim_cut,thvalues,L);
end

MatrixResults(15,:) = [Q_ATWTM3,Q_avg_ATWTM3,SAM_ATWTM3,ERGAS_ATWTM3,SCC_GT_ATWTM3];

%% MTF-GLP

cd GLP
t2=tic;
I_MTF_GLP = MTF_GLP(I_PAN,I_MS,sensor,im_tag,ratio);
% I_MTF_GLP = MTF_GLP_ECB(I_MS,I_PAN,ratio,[9 9],2.5,sensor,im_tag);
% I_MTF_GLP = MTF_GLP_CBD(I_MS,I_PAN,ratio,[55 55],-Inf,sensor,im_tag);
time_MTF_GLP = toc(t2);
fprintf('Elaboration time MTF-GLP: %.2f [sec]\n',time_MTF_GLP);
cd ..

[Q_avg_MTF_GLP, SAM_MTF_GLP, ERGAS_MTF_GLP, SCC_GT_MTF_GLP, Q_MTF_GLP] = indexes_evaluation(I_MTF_GLP,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);

if size(I_GT,3) == 4
    showImage4(I_MTF_GLP,printEPS,18,flag_cut_bounds,dim_cut,thvalues,L);
else
    showImage8(I_MTF_GLP,printEPS,18,flag_cut_bounds,dim_cut,thvalues,L);
end

MatrixResults(16,:) = [Q_MTF_GLP,Q_avg_MTF_GLP,SAM_MTF_GLP,ERGAS_MTF_GLP,SCC_GT_MTF_GLP];

%% MTF-GLP-HPM-PP

cd GLP
t2=tic;
I_MTF_GLP_HPM_PP = MTF_GLP_HPM_PP(I_PAN,I_MS_LR,sensor,im_tag,ratio);
time_MTF_GLP_HPM_PP = toc(t2);
fprintf('Elaboration time MTF-GLP-HPM-PP: %.2f [sec]\n',time_MTF_GLP_HPM_PP);
cd ..

[Q_avg_MTF_GLP_HPM_PP, SAM_MTF_GLP_HPM_PP, ERGAS_MTF_GLP_HPM_PP, SCC_GT_MTF_GLP_HPM_PP, Q_MTF_GLP_HPM_PP] = indexes_evaluation(I_MTF_GLP_HPM_PP,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
    
if size(I_GT,3) == 4
    showImage4(I_MTF_GLP_HPM_PP,printEPS,19,flag_cut_bounds,dim_cut,thvalues,L);
else
    showImage8(I_MTF_GLP_HPM_PP,printEPS,19,flag_cut_bounds,dim_cut,thvalues,L);
end

MatrixResults(17,:) = [Q_MTF_GLP_HPM_PP,Q_avg_MTF_GLP_HPM_PP,SAM_MTF_GLP_HPM_PP,ERGAS_MTF_GLP_HPM_PP,SCC_GT_MTF_GLP_HPM_PP];

%% MTF-GLP-HPM

cd GLP
t2=tic;
I_MTF_GLP_HPM = MTF_GLP_HPM(I_PAN,I_MS,sensor,im_tag,ratio);
time_MTF_GLP_HPM = toc(t2);
fprintf('Elaboration time MTF-GLP-HPM: %.2f [sec]\n',time_MTF_GLP_HPM);
cd ..

[Q_avg_MTF_GLP_HPM, SAM_MTF_GLP_HPM, ERGAS_MTF_GLP_HPM, SCC_GT_MTF_GLP_HPM, Q_MTF_GLP_HPM] = indexes_evaluation(I_MTF_GLP_HPM,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);

if size(I_GT,3) == 4    
    showImage4(I_MTF_GLP_HPM,printEPS,20,flag_cut_bounds,dim_cut,thvalues,L);    
else
    showImage8(I_MTF_GLP_HPM,printEPS,20,flag_cut_bounds,dim_cut,thvalues,L);
end

MatrixResults(18,:) = [Q_MTF_GLP_HPM,Q_avg_MTF_GLP_HPM,SAM_MTF_GLP_HPM,ERGAS_MTF_GLP_HPM,SCC_GT_MTF_GLP_HPM];

%% MTF-GLP-CBD

cd GS
t2=tic;

I_MTF_GLP_CBD = GS2_GLP(I_MS,I_PAN,ratio,sensor,im_tag);

time_MTF_GLP_CBD = toc(t2);
fprintf('Elaboration time MTF-GLP-CBD: %.2f [sec]\n',time_MTF_GLP_CBD);
cd ..

[Q_avg_MTF_GLP_CBD, SAM_MTF_GLP_CBD, ERGAS_MTF_GLP_CBD, SCC_GT_MTF_GLP_CBD, Q_MTF_GLP_CBD] = indexes_evaluation(I_MTF_GLP_CBD,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);

if size(I_GT,3) == 4
    showImage4(I_MTF_GLP_CBD,printEPS,21,flag_cut_bounds,dim_cut,thvalues,L);
else
    showImage8(I_MTF_GLP_CBD,printEPS,21,flag_cut_bounds,dim_cut,thvalues,L);
end

MatrixResults(19,:) = [Q_MTF_GLP_CBD,Q_avg_MTF_GLP_CBD,SAM_MTF_GLP_CBD,ERGAS_MTF_GLP_CBD,SCC_GT_MTF_GLP_CBD];

%% Print in LATEX

if size(I_GT,3) == 4
   matrix2latex(MatrixResults,'Real_Dataset.tex', 'rowLabels',[{'EXP'},{'PCA'},{'IHS'},{'Brovey'},{'BDSD'},{'GS'},{'GSA'},{'PRACS'},{'HPF'},{'SFIM'},{'Indusion'},{'ATWT'},{'AWLP'},...
        {'ATWT-M2'},{'ATWT-M3'},{'MTF-GLP'},{'MTF-GLP-HPM-PP'},{'MTF-GLP-HPM'},{'MTF-GLP-CBD'}],'columnLabels',[{'Q4'},{'Q'},{'SAM'},{'ERGAS'},{'SCC'}],'alignment','c','format', '%.4f');
else
   matrix2latex(MatrixResults,'Real_Dataset.tex', 'rowLabels',[{'EXP'},{'PCA'},{'IHS'},{'Brovey'},{'BDSD'},{'GS'},{'GSA'},{'PRACS'},{'HPF'},{'SFIM'},{'Indusion'},{'ATWT'},{'AWLP'},...
        {'ATWT-M2'},{'ATWT-M3'},{'MTF-GLP'},{'MTF-GLP-HPM-PP'},{'MTF-GLP-HPM'},{'MTF-GLP-CBD'}],'columnLabels',[{'Q8'},{'Q'},{'SAM'},{'ERGAS'},{'SCC'}],'alignment','c','format', '%.4f'); 
end

%% View All

MatrixImage(:,:,:,1) = I_GT;
MatrixImage(:,:,:,2) = I_MS;
MatrixImage(:,:,:,3) = I_PCA;
MatrixImage(:,:,:,4) = I_IHS;
MatrixImage(:,:,:,5) = I_Brovey;
MatrixImage(:,:,:,6) = I_BDSD;
MatrixImage(:,:,:,7) = I_GS;
MatrixImage(:,:,:,8) = I_GSA;
MatrixImage(:,:,:,9) = I_PRACS;
MatrixImage(:,:,:,10) = I_HPF;
MatrixImage(:,:,:,11) = I_SFIM;
MatrixImage(:,:,:,12) = I_Indusion;
MatrixImage(:,:,:,13) = I_ATWT;
MatrixImage(:,:,:,14) = I_AWLP;
MatrixImage(:,:,:,15) = I_ATWTM2;
MatrixImage(:,:,:,16) = I_ATWTM3;
MatrixImage(:,:,:,17) = I_MTF_GLP;
MatrixImage(:,:,:,18) = I_MTF_GLP_HPM_PP;
MatrixImage(:,:,:,19) = I_MTF_GLP_HPM;
MatrixImage(:,:,:,20) = I_MTF_GLP_CBD;

if size(I_GT,3) == 4
    vect_index_RGB = [3,2,1];
else
    vect_index_RGB = [5,3,1];
end

titleImages = {'GT','EXP','PCA','IHS','Brovey','BDSD','GS','GSA','PRACS','HPF','SFIM','Indusion','ATWT','AWLP','ATWT M2','ATWT M3','MTF GLP','MTF GLP HPM PP','MTF GLP HPM','MTF GLP CBD'};

figure, showImagesAll(MatrixImage,titleImages,vect_index_RGB,flag_cut_bounds,dim_cut,0);