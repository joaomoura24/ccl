% Get results
load('results.mat');

% Get image
cenas = W_corr_mean./max(max(W_corr_mean));

figure();
imagesc(cenas); colorbar; colormap jet;


% cenas = im2single(cenas);
% cenas = imgaussfilt(cenas,1);
% edgeThreshold = 0.4;
% amount = 0.5;
% cenas = localcontrast(cenas, edgeThreshold, amount);
% % %cenas = imsharpen(cenas);
% % cenas = cenas./max(max(cenas));
% % cenas = imadjust(cenas);
% % %cenas = imgaussfilt(cenas);
% % %cenas = imsharpen(cenas);
% 
% figure();
% imagesc(cenas); colorbar; colormap jet;
% 
% 
% 
% %
% %error('stop here');
% % Get filter
% N = length(cenas);
% n = -floor(N/2):1:ceil(N/2)-1;
% [x,y] = meshgrid(n,n);
% c = (y == 0 & x ~= 0);
% % | (x == 0 & y ~= 0 & abs(y)<kk);
% %c = (x == 0 & abs(y)>kk);
% %c = abs(x) > 1;
% %c = not(c);
% 
% 
% %c = z < kk;
% 
% 
% 
% % fftshift(
% 
% 
% F = fftshift(fft2(cenas));
% Fmag = abs(F);
% Fang = angle(F);
% FF = ifftshift((c.*Fmag).*exp(i*Fang));
% iF = real(ifft2(FF));
% 
% %abs(fftA).*exp(i*angle(fftB));
% 
% figure();
% subplot(2,1,1); imagesc(log(c.*Fmag)); colorbar; colormap jet;
% subplot(2,1,2); imagesc(Fang); colorbar; colormap jet;
% 
% 
% figure();
% subplot(2,1,1); imagesc(cenas); colorbar; colormap jet;
% subplot(2,1,2); imagesc(iF); colorbar; colormap jet;
% 
% 
% figure();
% subplot(2,1,1); plot(iF(1,:));
% subplot(2,1,2); plot(abs(diff(iF(1,:))));
% 
% 
% %error('stop here');
% % Transform it into image
% I = mat2gray(iF);
% figure(); imshow(I);
% 
% 
% %// Filter the image, using itself as a guide
% %I = imguidedfilter(I);
% %I = imsharpen(I);
% 
% 
% BW = edge(I,'canny');
% figure(); imshow(BW);
% 
% error('stop here');
% % 
% % %F = fft(cenas,[],2);
% % 
% % 
% % 
% % figure(); imagesc(iF);
% % colorbar; colormap jet;
% % 
% % error('stop here');
% % % 
% % % 
% % % res = cenas(10,:);
% % % figure(); plot(res);
% % % f = fft2(res);
% % % f_val = abs(f);
% % % figure(); plot(f_val);
% % % f_ang = unwrap(angle(f));
% % % figure(); plot(f_ang);
% % % 
% % % 
% % % error('stop here');
% % %W_corr_mean = (W_corr_mean).^W_corr_mean;
% % 
% % I = mat2gray(W_corr_mean);
% % figure(); 
% % imshow(I);





I = mat2gray(cenas);
I_original = I;
figure();
subplot(2,4,1); imshow(I);


SE = strel('square',20);
I = imclose(I, SE);
subplot(2,4,2); imshow(I);

T1 = imbinarize(I,0.04);
subplot(2,4,3); imshow(T1);

T2 = imbinarize(I,'adaptive','ForegroundPolarity','dark','Sensitivity',0.1);
subplot(2,4,4); imshow(T2);

T3 = not(T1).*not(T2);
subplot(2,4,5); imshow(T3);

SE = strel('square',10);
BW = imopen(T3,SE);
subplot(2,4,6); imshow(BW);

SE = strel('square',10);
BW = imerode(BW,SE);
subplot(2,4,6); imshow(BW);


figure();
imshow(I_original);

load('ref_p.mat');
hold on;
ppi = floor(pi./50);
ppf = floor(pf./50);
dpp = ppf - ppi;
for idx=1:length(pi)
    rectangle('Position', [ppi(idx) ppi(idx) dpp(idx) dpp(idx)], ...
        'Linewidth', 1, 'EdgeColor', 'r', 'lineStyle', '--');
end

figure();
imshow(BW);

load('ref_p.mat');
hold on;
ppi = floor(pi./50);
ppf = floor(pf./50);
dpp = ppf - ppi;
for idx=1:length(pi)
    rectangle('Position', [ppi(idx) ppi(idx) dpp(idx) dpp(idx)], ...
        'Linewidth', 1, 'EdgeColor', 'r', 'lineStyle', '--');
end


error('stop here');



%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
% Prepare image and find regions:
A = cenas;
kk = 0.03;
A(A<kk) = 0;
A(A>kk) = 1;
figure();
imagesc(A); colorbar; colormap jet;

% Transform it into image
I = mat2gray(A);
figure(); 
imshow(I);

% Binarize image:
I_bw = imbinarize(I);
figure();
imshow(I_bw);

% Invert image
I_wb = imcomplement(I_bw);
figure(); imshow(I_wb);

% Erode image
I_erode = imerode(I_wb,ones(15));
figure(); imshow(I_erode);

% dilate image
I_dilate = imdilate(I_erode,ones(15));
figure(); imshow(I_dilate);

% find regions
stats = regionprops(I_dilate,'all');
hold on;
r = cell(1,numel(stats));
for i=1:numel(stats)
    r{i} = rectangle('Position', stats(i).BoundingBox, ...
        'Linewidth', 3, 'EdgeColor', 'r', 'lineStyle', '--');
end