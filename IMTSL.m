%% INCREMENTAL TENSOR SUBSPACE LEARNING (VERSION 2)
%%% BMC
close all; clear all; clc;

%%% TOOLBOXES
addpath('libs/tensor_toolbox_2.5');
addpath('libs/IncPACK2');
addpath('libs/features/lbp/LBP_toolbox');
% addpath('libs/tensorlab'); % for slice3(...)

%% LOAD DATASET
filename = 'dataset/BMC2012/Evaluation Phase/Street/112.avi';
xyloObj = VideoReader(filename); clear filename;
nFrames = xyloObj.NumberOfFrames;
vidHeight = xyloObj.Height;
vidWidth = xyloObj.Width;
% for k = 1 : nFrames
%   I = read(xyloObj,k);
%   I = flipdim(I,1);
%   imshow(I,[],'InitialMagnification','fit');
%   disp(k);
%   pause(0.1);
% end

%% INCREMENTAL TENSOR SUBSPACE LEARNING
clc;
tic;
block_size = 25;
%block_size = 10;
A0 = zeros(vidHeight,vidWidth,3,block_size); % for rgb
%A0_gray = zeros(vidHeight,vidWidth,block_size); % for gray
%A0 = zeros(vidHeight,vidWidth,block_size); % for gray
displog('Building initial block video');
for k = 1 : block_size
  Irgb = read(xyloObj,k);
  Irgb = flip(Irgb,1);
  A0(:,:,:,k) = Irgb; % for rgb
  %A0_gray(:,:,k) = rgb2gray(Irgb);
  % A0(:,:,k) = I; % for gray
end
[m,n,~,p] = size(A0); % for rgb
%[m,n,p] = size(A0); % for gray

%% Build Feature Tensor Model
displog('Performing feature extraction and building feature tensor model');
F0 = perform_feature_extraction_rgb(A0); % for rgb
%F0 = perform_feature_extraction_gray(A0); % for gray
T0 = tensor(F0);
F0_f1 = F0(:,:,1); % reg
F0_f2 = F0(:,:,2); % green
F0_f3 = F0(:,:,3); % blue
F0_f4 = F0(:,:,4); % gray
F0_f5 = F0(:,:,5); % lbp
F0_f6 = F0(:,:,6); % mag
F0_f7 = F0(:,:,7); % gx
F0_f8 = F0(:,:,8); % gy

% figure;
% subplot(2,2,1), imagesc(F0_f1);
% subplot(2,2,2), imagesc(F0_f2);
% subplot(2,2,3), imagesc(F0_f3);
% subplot(2,2,4), imagesc(F0_f4);

%%% show tensor model (slice3 surf3 voxel3)
% slice3(A0), colormap('gray'); % only if A0 is 3D
% slice3(F0), colormap('gray');
% slice3(A0_gray), colormap('gray');

%% Unfolding 3rd-order tensor into mode-(1,2,3) and convert it to matrix
T01tm = tenmat(T0,1); T01 = double(T01tm);
T02tm = tenmat(T0,2); T02 = double(T02tm);
T03tm = tenmat(T0,3); T03 = double(T03tm);

%%% Perform initial SVD in the mode-1
displog('Performing initial SVD in the mode-1');
r1 = 1; % temporal/values rank
[T01_U,T01_S,T01_V] = svds(T01,r1);
T01_hat = (T01_U*T01_S*T01_V'); % norm(T01-T01_hat)
T0_hat1 = double(tensor(tenmat(T01_hat,T01tm.rdims,T01tm.cdims,T01tm.tsize)));
F0_hat1_f1 = T0_hat1(:,:,1);
F0_hat1_f2 = T0_hat1(:,:,2);
F0_hat1_f3 = T0_hat1(:,:,3);
F0_hat1_f4 = T0_hat1(:,:,4);
F0_hat1_f5 = T0_hat1(:,:,5);
F0_hat1_f6 = T0_hat1(:,:,6);
F0_hat1_f7 = T0_hat1(:,:,7);
F0_hat1_f8 = T0_hat1(:,:,8);

% figure;
% subplot(2,2,1), imagesc(F0_hat1_f1);
% subplot(2,2,2), imagesc(F0_hat1_f2);
% subplot(2,2,3), imagesc(F0_hat1_f3);
% subplot(2,2,4), imagesc(F0_hat1_f4);
% figure; showResults(F0_f1',F0_hat1_f1',[],[],p,m,n);
% figure; showResults(F0_f2',F0_hat1_f2',[],[],p,m,n);
% figure; showResults(F0_f3',F0_hat1_f3',[],[],p,m,n);
% figure; showResults(F0_f4',F0_hat1_f4',[],[],p,m,n);

%%% Perform initial SVD in the mode-2
% r2 = 4; % feature/pixel rank (# of features)
% [T02_U,T02_S,T02_V] = svds(T02,r2);
% T02_hat = (T02_U*T02_S*T02_V');
% T0_hat2 = double(tensor(tenmat(T02_hat,T02tm.rdims,T02tm.cdims,T02tm.tsize)));
% F0_hat2_f1 = T0_hat2(:,:,1);
% F0_hat2_f2 = T0_hat2(:,:,2);
% F0_hat2_f3 = T0_hat2(:,:,3);
% F0_hat2_f4 = T0_hat2(:,:,4);

% figure;
% subplot(1,2,1), imagesc(F0_hat2_f1);
% subplot(1,2,2), imagesc(F0_hat2_f2);
% figure; showResults(F0_f1',F0_hat2_f1',[],[],p,m,n);
% figure; showResults(F0_f2',F0_hat2_f2',[],[],p,m,n);
% figure; showResults(F0_f3',F0_hat2_f3',[],[],p,m,n);
% figure; showResults(F0_f4',F0_hat2_f4',[],[],p,m,n);

%%% Perform initial SVD in the mode-3
% r3 = 2; % pixel/feature rank % don't build static bg model
% [T03_U,T03_S,T03_V] = svds(T03,r3);
% T03_hat = (T03_U*T03_S*T03_V');
% T0_hat3 = double(tensor(tenmat(T03_hat,T03tm.rdims,T03tm.cdims,T03tm.tsize)));
% F0_hat3_f1 = T0_hat3(:,:,1);
% F0_hat3_f2 = T0_hat3(:,:,2);

% figure;
% subplot(1,2,1), imagesc(F0_hat3_f1);
% subplot(1,2,2), imagesc(F0_hat3_f2);
% figure; showResults(F0_f1',F0_hat3_f1',[],[],p,m,n);
% figure; showResults(F0_f2',F0_hat3_f2',[],[],p,m,n);

%% Incremental Tensor Learning
%%% When new frame arrives, add it into the tensor and drop the last frame
firstFrame = 1;
%for i = 1:nFrames
%for i = block_size+1:nFrames %size(video,3)
for i = 200:nFrames %size(video,3) % only for demonstration
  if(firstFrame == 1)
    A = A0;
    %A_gray = A0_gray;
    F = F0;
    firstFrame = 0;
  end
  
  displog(['Processing frame ' num2str(i)]);
  
  % Get new frame
  % I = video(:,:,i); % for gray
  Irgb = read(xyloObj,i);
  Irgb = flip(Irgb,1);
  I = rgb2gray(Irgb);
  
  % Update video block
  displog('Updating video block');
  % A(:,:,end+1) = I;
  % A = A(:,:,2:end);
  A(:,:,:,end+1) = Irgb;
  A = A(:,:,:,2:end);
  %A_gray(:,:,end+1) = I;
  %A_gray = A_gray(:,:,2:end);
  
  % Feature Extraction
  displog('Performing feature extraction');
  Ir = Irgb(:,:,1); % imshow(Ir,[],'InitialMagnification','fit');
  Ig = Irgb(:,:,2); % imshow(Ig,[],'InitialMagnification','fit');
  Ib = Irgb(:,:,3); % imshow(Ib,[],'InitialMagnification','fit');
  %Ilbp = lbp(I, 0);
  Ilbp = lbp_median(I, 0, 1, 8);
  [Gx, Gy] = imgradientxy(I);
  [Gmag, ~] = imgradient(Gx, Gy);
  %[Gmag, Gdir] = imgradient(Gx, Gy);

  vIr = reshape(Ir,[],1);
  vIg = reshape(Ig,[],1);
  vIb = reshape(Ib,[],1);
  vI = reshape(I,[],1);
  vIlbp = reshape(Ilbp,[],1);
  vImag = reshape(Gmag,[],1);
  vIgx = reshape(Gx,[],1);
  vIgy = reshape(Gy,[],1);
  %vIdir = reshape(Gdir,[],1);
    
  % Update tensor model
  displog('Updating feature tensor model');
  F(end+1,:,1) = vIr;
  F(end,:,2) = vIg;
  F(end,:,3) = vIb;
  F(end,:,4) = vI;
  F(end,:,5) = vIlbp;
  F(end,:,6) = vImag;
  F(end,:,7) = vIgx;
  F(end,:,8) = vIgy;
  %F(end,:,9) = vIdir;
  F = F(2:end,:,:);
  
  T = tensor(F);
  F_f1 = F(:,:,1);
  F_f2 = F(:,:,2);
  F_f3 = F(:,:,3);
  F_f4 = F(:,:,4);
  F_f5 = F(:,:,5);
  F_f6 = F(:,:,6);
  F_f7 = F(:,:,7);
  F_f8 = F(:,:,8);
  
  % figure;
  % subplot(2,2,1), imagesc(F0_f1), pause(0.5), imagesc(F_f1);
  % subplot(2,2,2), imagesc(F0_f2), pause(0.5), imagesc(F_f2);
  % subplot(2,2,3), imagesc(F0_f3), pause(0.5), imagesc(F_f3);
  % subplot(2,2,4), imagesc(F0_f4), pause(0.5), imagesc(F_f4);
  
  %%% show tensor model (slice3 surf3 voxel3)
  % slice3(A), colormap('gray');
  % slice3(F), colormap('gray');
  % slice3(A_gray), colormap('gray');
  
  %%% Unfolding 3rd-order tensor into mode-(1,2,3) and convert it to matrix
  T1tm = tenmat(T,1); T1 = double(T1tm);
  T2tm = tenmat(T,2); T2 = double(T2tm);
  T3tm = tenmat(T,3); T3 = double(T3tm);
  
  %%% Perform incremental SVD in the mode-1
  displog('Performing incremental SVD in the mode-1');
  T1opts.U = T01_U;
  T1opts.S = T01_S;
  T1opts.V = T01_V;
  T1opts.debug = 0;
  [T1_U,T1_S,T1_V] = seqkl(T1,r1,0.01,T1opts);
  T1_hat = (T1_U*T1_S*T1_V');
  T_hat1 = double(tensor(tenmat(T1_hat,T1tm.rdims,T1tm.cdims,T1tm.tsize)));
  F_hat1_f1 = T_hat1(:,:,1);
  F_hat1_f2 = T_hat1(:,:,2);
  F_hat1_f3 = T_hat1(:,:,3);
  F_hat1_f4 = T_hat1(:,:,4);
  F_hat1_f5 = T_hat1(:,:,5);
  F_hat1_f6 = T_hat1(:,:,6);
  F_hat1_f7 = T_hat1(:,:,7);
  F_hat1_f8 = T_hat1(:,:,8);

  % figure;
  % subplot(2,2,1), imagesc(F0_hat1_f1), pause(0.5), imagesc(F_hat1_f1);
  % subplot(2,2,2), imagesc(F0_hat1_f2), pause(0.5), imagesc(F_hat1_f2);
  % subplot(2,2,3), imagesc(F0_hat1_f3), pause(0.5), imagesc(F_hat1_f3);
  % subplot(2,2,4), imagesc(F0_hat1_f4), pause(0.5), imagesc(F_hat1_f4);
  % figure; showResults(F_f1',F_hat1_f1',[],[],p,m,n);
  % figure; showResults(F_f2',F_hat1_f2',[],[],p,m,n);
  % figure; showResults(F_f3',F_hat1_f3',[],[],p,m,n);
  % figure; showResults(F_f4',F_hat1_f4',[],[],p,m,n);
  
  % updates the old mode-1 svd
  T01_U = T1_U;
  T01_S = T1_S;
  T01_V = T1_V;
  
  %%% Perform incremental SVD in the mode-2
%   T2opts.U = T02_U;
%   T2opts.S = T02_S;
%   T2opts.V = T02_V;
%   T2opts.debug = 0;
%   [T2_U,T2_S,T2_V] = seqkl(T2,r2,0.01,T2opts);
%   T2_hat = (T2_U*T2_S*T2_V');
%   T_hat2 = double(tensor(tenmat(T2_hat,T2tm.rdims,T2tm.cdims,T2tm.tsize)));
%   F_hat2_f1 = T_hat2(:,:,1);
%   F_hat2_f2 = T_hat2(:,:,2);
%   F_hat2_f3 = T_hat2(:,:,3);
%   F_hat2_f4 = T_hat2(:,:,4);

  % figure;
  % subplot(2,2,1), imagesc(F0_hat2_f1), pause(0.5), imagesc(F_hat2_f1);
  % subplot(2,2,2), imagesc(F0_hat2_f2), pause(0.5), imagesc(F_hat2_f2);
  % subplot(2,2,3), imagesc(F0_hat2_f3), pause(0.5), imagesc(F_hat2_f3);
  % subplot(2,2,4), imagesc(F0_hat2_f4), pause(0.5), imagesc(F_hat2_f4);
  % figure; showResults(F_f1',F_hat2_f1',[],[],p,m,n);
  % figure; showResults(F_f2',F_hat2_f2',[],[],p,m,n);
  % figure; showResults(F_f3',F_hat2_f3',[],[],p,m,n);
  % figure; showResults(F_f4',F_hat2_f4',[],[],p,m,n);
  
  % updates the old mode-2 svd
%   T02_U = T2_U;
%   T02_S = T2_S;
%   T02_V = T2_V;
  
  %%% Perform incremental SVD in the mode-3
%   T3opts.U = T03_U;
%   T3opts.S = T03_S;
%   T3opts.V = T03_V;
%   T3opts.debug = 0;
%   [T3_U,T3_S,T3_V] = seqkl(T3,r3,0.01,T3opts);
%   T3_hat = (T3_U*T3_S*T3_V');
%   T_hat3 = double(tensor(tenmat(T3_hat,T3tm.rdims,T3tm.cdims,T3tm.tsize)));
%   F_hat3_f1 = T_hat3(:,:,1);
%   F_hat3_f2 = T_hat3(:,:,2);

  % figure;
  % subplot(1,2,1), imagesc(F0_hat3_f1), pause(0.5), imagesc(F_hat3_f1);
  % subplot(1,2,2), imagesc(F0_hat3_f2), pause(0.5), imagesc(F_hat3_f2);
  % figure; showResults(F_f1',F_hat3_f1',[],[],p,m,n);
  % figure; showResults(F_f2',F_hat3_f2',[],[],p,m,n);
  
  %%% Foreground Detection
  displog('Calculating similarity measures');
  
  I_F_hat1_f1 = reshape(F_hat1_f1(end,:),[m n]); % imshow(I_F_hat1_f1,[],'InitialMagnification','fit');
  I_F_hat1_f2 = reshape(F_hat1_f2(end,:),[m n]); % imshow(I_F_hat1_f2,[],'InitialMagnification','fit');
  I_F_hat1_f3 = reshape(F_hat1_f3(end,:),[m n]); % imshow(I_F_hat1_f3,[],'InitialMagnification','fit');
  I_F_hat1_f4 = reshape(F_hat1_f4(end,:),[m n]); % imshow(I_F_hat1_f4,[],'InitialMagnification','fit');
  I_F_hat1_f5 = reshape(F_hat1_f5(end,:),[m n]); % imshow(I_F_hat1_f5,[],'InitialMagnification','fit');
  I_F_hat1_f6 = reshape(F_hat1_f6(end,:),[m n]); % imshow(I_F_hat1_f6,[],'InitialMagnification','fit');
  I_F_hat1_f7 = reshape(F_hat1_f7(end,:),[m n]); % imshow(I_F_hat1_f7,[],'InitialMagnification','fit');
  I_F_hat1_f8 = reshape(F_hat1_f8(end,:),[m n]); % imshow(I_F_hat1_f8,[],'InitialMagnification','fit');
  
  [Sim_F_hat1_f1] = compute_similarity2(Ir,I_F_hat1_f1); % imshow(Sim_F_hat1_f1,[],'InitialMagnification','fit');
  [Sim_F_hat1_f2] = compute_similarity2(Ig,I_F_hat1_f2); % imshow(Sim_F_hat1_f2,[],'InitialMagnification','fit');
  [Sim_F_hat1_f3] = compute_similarity2(Ib,I_F_hat1_f3); % imshow(Sim_F_hat1_f3,[],'InitialMagnification','fit');
  [Sim_F_hat1_f4] = compute_similarity2(I,I_F_hat1_f4); % imshow(Sim_F_hat1_f4,[],'InitialMagnification','fit');
  [Sim_F_hat1_f5] = compute_similarity2(Ilbp,I_F_hat1_f5); % imshow(Sim_F_hat1_f5,[],'InitialMagnification','fit');
  %[Sim_F_hat1_f6] = compute_similarity2(uint8(Gmag),uint8(I_F_hat1_f6)); % imshow(Sim_F_hat1_f6,[],'InitialMagnification','fit');
  [Sim_F_hat1_f6] = compute_similarity2((Gmag),(I_F_hat1_f6)); % imshow(Sim_F_hat1_f6,[],'InitialMagnification','fit');
  [Sim_F_hat1_f7] = compute_similarity3(Gx,I_F_hat1_f7); % imshow(Sim_F_hat1_f7,[],'InitialMagnification','fit');
  [Sim_F_hat1_f8] = compute_similarity3(Gy,I_F_hat1_f8); % imshow(Sim_F_hat1_f8,[],'InitialMagnification','fit');
  
  %Sim_F_hat1_f5 = medfilt2(Sim_F_hat1_f5, [5 5]);
  %Sim_F_hat1_f5 = imfilter(Sim_F_hat1_f5, fspecial('average'));
  %Sim_F_hat1_f5 = entropyfilt(Sim_F_hat1_f5); 
  
%   I_F_hat2_f1 = reshape(F_hat2_f1(end,:),[m n]); % imshow(I_F_hat2_f1,[],'InitialMagnification','fit');
%   I_F_hat2_f2 = reshape(F_hat2_f2(end,:),[m n]); % imshow(I_F_hat2_f2,[],'InitialMagnification','fit');
%   I_F_hat2_f3 = reshape(F_hat2_f3(end,:),[m n]); % imshow(I_F_hat2_f3,[],'InitialMagnification','fit');
%   I_F_hat2_f4 = reshape(F_hat2_f4(end,:),[m n]); % imshow(I_F_hat2_f4,[],'InitialMagnification','fit');
  
%   [Sim_F_hat2_f1] = compute_similarity2(Ir,I_F_hat2_f1); % imshow(Sim_F_hat2_f1,[],'InitialMagnification','fit');
%   [Sim_F_hat2_f2] = compute_similarity2(Ig,I_F_hat2_f2); % imshow(Sim_F_hat2_f2,[],'InitialMagnification','fit');
%   [Sim_F_hat2_f3] = compute_similarity2(Ib,I_F_hat2_f3); % imshow(Sim_F_hat2_f3,[],'InitialMagnification','fit');
%   [Sim_F_hat2_f4] = compute_similarity2(I,I_F_hat2_f4); % imshow(Sim_F_hat2_f4,[],'InitialMagnification','fit');
  
%   [Sim_F_hat1_f1] = compute_similarity(I,I_F_hat1_f1); % imshow(Sim_F_hat1_f1,[],'InitialMagnification','fit');
%   [Sim_F_hat1_f2] = compute_similarity(Ilbp,I_F_hat1_f2); % imshow(Sim_F_hat1_f2,[],'InitialMagnification','fit');
%   [Sim_F_hat1_f3] = compute_similarity(Gmag,I_F_hat1_f3); % imshow(Sim_F_hat1_f3,[],'InitialMagnification','fit');
%   [Sim_F_hat1_f4] = compute_similarity(Gdir,I_F_hat1_f4); % imshow(Sim_F_hat1_f4,[],'InitialMagnification','fit');
%   
%   [Sim_F_hat2_f1] = compute_similarity(I,I_F_hat2_f1); % imshow(Sim_F_hat2_f1,[],'InitialMagnification','fit');
%   [Sim_F_hat2_f2] = compute_similarity(Ilbp,I_F_hat2_f2); % imshow(Sim_F_hat2_f2,[],'InitialMagnification','fit');
%   [Sim_F_hat2_f3] = compute_similarity(Gmag,I_F_hat2_f3); % imshow(Sim_F_hat2_f3,[],'InitialMagnification','fit');
%   [Sim_F_hat2_f4] = compute_similarity(Gdir,I_F_hat2_f4); % imshow(Sim_F_hat2_f4,[],'InitialMagnification','fit');
  
  %%% Perceptron
  displog('Performing foreground detection');
  w1 = 0.125; w2 = 0.125; w3 = 0.125; w4 = 0.225; w5 = 0.0250; w6 = 0.125; w7 = 0.125; w8 = 0.125;
  wsum = w1+w2+w3+w4+w5+w6+w7+w8;
  x1w1 = Sim_F_hat1_f1 * w1;
  x2w2 = Sim_F_hat1_f2 * w2;
  x3w3 = Sim_F_hat1_f3 * w3;
  x4w4 = Sim_F_hat1_f4 * w4;
  x5w5 = Sim_F_hat1_f5 * w5;
  x6w6 = Sim_F_hat1_f6 * w6;
  x7w7 = Sim_F_hat1_f7 * w7;
  x8w8 = Sim_F_hat1_f8 * w8;
  sf = x1w1+x2w2+x3w3+x4w4+x5w5+x6w6+x7w7+x8w8;
  sf = sf/wsum;
  out = (sf <= 0.5);
  %imshow(out,[],'InitialMagnification','fit');
  %out = medfilt2(out, [5 5]);
  %imwrite(Irgb,'frame200.png');
  
  %%%
  %w1 = 0.5; w2 = 0.5;
  %w1 = 0.3; w2 = 0.3; w3 = 0.3; w4 = 0.3;
  %w1 = 0.35; w2 = 0.35; w3 = 0.15; w4 = 0.15;
  %x1w1 = Sim_F_hat1_f1 * w1;
  %x2w2 = Sim_F_hat2_f1 * w2;
  %x3w3 = Sim_F_hat1_f2 * w3;
  %x4w4 = Sim_F_hat2_f2 * w4;
  %x3w3 = Sim_F_hat1_f3 * w3;
  %x4w4 = Sim_F_hat2_f3 * w4;
  %sf = x1w1+x2w2;
  %sf = x1w1+x2w2+x3w3+x4w4;
  %sf = medfilt2(sf, [3 3]);
  %out = (sf < 0.90);
  %out = (sf < 0.70);
  %out = (sf < 0.75);
  
  % figure; imagesc(sf);
  subplot(1,2,1), imshow(Irgb,[],'InitialMagnification','fit');
  subplot(1,2,2), imshow(out,[],'InitialMagnification','fit');
  
  %out_filename = ['output/mask/' num2str(i) '.png'];
  %displog(['Saving results in: ' out_filename]);
  %imwrite(out,out_filename);
  
  pause(0.1);
  break;
end

disp('Finished');
toc

%% BUILD FOREGROUND VIDEO
clear; clc;

start_idx = 1;
stop_idx = 1498;

movobj_I(1:(stop_idx-start_idx+1)) = struct('cdata', [], 'colormap', []);

k = 1;
for i = start_idx:stop_idx
  filename = ['output/mask/' num2str(i) '.png'];
  filename2 = ['output/mask/resize/' num2str(i) '.png'];
  disp(['Reading: ' filename]);
  
  %I = imread(filename);
  I = uint8(imread(filename))*255; % imagesc(I);
  %I = medfilt2(I);
  I = imresize(I, 4); % imagesc(I);
  %I = medfilt2(I);
  Irgb = repmat(I,[1 1 3]); % imagesc(Irgb);
  
  movobj_I(k).cdata = Irgb;
  imwrite(I,filename2);
  k = k + 1;
end

%% SAVE FOREGROUND VIDEO
out_filename = ['output/mask.avi'];
disp(['Saving results in: ' out_filename]);
movie2avi(movobj_I, out_filename, 'compression', 'None');
disp('OK');
