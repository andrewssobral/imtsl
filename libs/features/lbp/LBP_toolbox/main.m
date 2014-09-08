addpath('features/lbp/LBP_toolbox');
clc;
Ilbp = lbp(I, 0);
Ilbp_inter = lbp_inter(I, 0, 1, 8);
Ilbp_median = lbp_median(I, 0, 1, 8);
Ilbp_ni = lbp_ni(I, 0, 1, 8);
Ilbp_num = lbp_num(I, 1, 8);
Ilbp_uni = lbp_uni(I, 1, 8);

figure;
subplot(2,3,1), imshow(Ilbp,[],'InitialMagnification','fit');
subplot(2,3,2), imshow(Ilbp_inter,[],'InitialMagnification','fit');
subplot(2,3,3), imshow(Ilbp_median,[],'InitialMagnification','fit');
subplot(2,3,4), imshow(Ilbp_ni,[],'InitialMagnification','fit');
subplot(2,3,5), imshow(Ilbp_num,[],'InitialMagnification','fit');
subplot(2,3,6), imshow(Ilbp_uni,[],'InitialMagnification','fit');

[s1] = compute_similarity2(Ilbp,Ilbp_median);

figure;
imshow(s1,[],'InitialMagnification','fit');