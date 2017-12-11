%close all;
% Get results
load('results_test_paper.mat');

% Get image
cenas = W_corr_mean./max(max(W_corr_mean));
figure();
imagesc(cenas); colorbar; colormap parula;
xlabel('Number of the window used to compute mean residual');
ylabel('Number of the window used to estimate constraint parameters');



load('ref_r_paper.mat');
hold on;
for idx=1:length(ri)
    ppi = floor(ri./50)-1;
    ppf = floor(rf./50)-1;
    dpp = ppf - ppi;
    rectangle('Position', [ppi(idx) ppi(idx) dpp(idx) dpp(idx)], ...
        'Linewidth', 1, 'EdgeColor', 'r', 'lineStyle', '--');
end

error('stop here');


cenas = medfilt2(cenas);




figure();
imagesc(cenas); colorbar; colormap jet;

error('stop here');


% Get filter
N = length(cenas);
n = -floor(N/2):1:ceil(N/2)-1;
[x,y] = meshgrid(n,n);
kk = 2;
kk2 = 30;
c = (y == 0) | x == 0;% | (y == 0 & x == 0);% | (x == 0 & abs(y) > kk);
%c = (y == 0 & x ~= 0);
F = fftshift(fft2(cenas));
Fmag = abs(F);
Fang = angle(F);
FF = ifftshift((c.*Fmag).*exp(1i*Fang));
iF = real(ifft2(FF));


figure();
subplot(2,1,1); imagesc(log(c.*Fmag)); colorbar; colormap jet;
subplot(2,1,2); imagesc(Fang); colorbar; colormap jet;


figure();
subplot(2,1,1); imagesc(cenas); colorbar; colormap jet;
subplot(2,1,2); imagesc(iF); colorbar; colormap jet;



figure(); plot(iF(1,:));



error('stop here');


newcenas = iF;

edgeThreshold = 0.0;
amount = 0.5;
newcenas = im2single(newcenas);
newcenas = localcontrast(newcenas, edgeThreshold, amount);
figure();
imagesc(newcenas); colorbar; colormap jet;



I = mat2gray(newcenas);
figure(); imshow(I);

BW = edge(I,'Sobel',0.005,'vertical');
figure(); imshow(BW);


