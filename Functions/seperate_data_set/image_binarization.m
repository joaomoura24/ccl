close all;
% Get results
load('results.mat');

% Plot mean error
figure();
imagesc(W_corr_mean); colorbar; colormap jet;

% Transform it into image
I = mat2gray(W_corr_mean);
figure(); 
imshow(I);

% Binarize image:
I_bw = imbinarize(I, 0.01);
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
for i=1:numel(stats)
    rectangle('Position', stats(i).BoundingBox, ...
        'Linewidth', 3, 'EdgeColor', 'r', 'lineStyle', '--');
end