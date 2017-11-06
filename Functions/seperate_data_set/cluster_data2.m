%% Auxiliar functions:
%--------------------------------------------------------------------------
getWFileName = @(w,i) strcat('W_A/w',num2str(w),'i',num2str(i),'.mat');
dot_prod = @(a,b) sum(bsxfun(@times,bsxfun(@minus,a,b),bsxfun(@minus,a,b)),1);
%--------------------------------------------------------------------------

%% User input
%--------------------------------------------------------------------------
window_size = 4000;
increment = 4000;
%--------------------------------------------------------------------------

%% Load estimated gains W
%--------------------------------------------------------------------------
file_name = getWFileName(window_size, increment);
load(file_name);
%--------------------------------------------------------------------------
% load('W_true_A.mat');
% load('H_cell_A.mat'); Hw = H;
%% Compute Correlation Matrix between W of different windows
%--------------------------------------------------------------------------
Nwindows = length(W_hat_vec);
W_corr_mean = zeros(Nwindows);
W_corr_std = zeros(Nwindows);
tic;
count = 0;
Ncount = Nwindows^2;
for idx_row=1:Nwindows
    for idx_col=1:Nwindows
        prov = W_hat{idx_row}*Hw{idx_col};
        W_corr_mean(idx_row, idx_col) = norm(mean(prov,2));
        W_corr_std(idx_row, idx_col) = norm(std(prov,1,2));
        disp((count/Ncount)*100);
        count = count + 1;
    end
end
toc
%--------------------------------------------------------------------------

%% Display correlation Matrix
%--------------------------------------------------------------------------
figure();
imagesc(W_corr_mean); colorbar; colormap jet;
%caxis([0 0.0005]);
figure();
imagesc(W_corr_std); colorbar; colormap jet;
%caxis([0 0.005]);
%--------------------------------------------------------------------------