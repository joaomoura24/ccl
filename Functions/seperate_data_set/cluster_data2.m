%% Auxiliar functions:
%--------------------------------------------------------------------------
getWFileName = @(w,i) strcat('W_test/w',num2str(w),'i',num2str(i),'.mat');
dot_prod = @(a,b) sum(bsxfun(@times,bsxfun(@minus,a,b),bsxfun(@minus,a,b)),1);
%--------------------------------------------------------------------------

%% User input
%--------------------------------------------------------------------------
window_size = 400;
increment = 50;
%--------------------------------------------------------------------------

%% Load estimated gains W
%--------------------------------------------------------------------------
file_name = getWFileName(window_size, increment);
load(file_name);
%--------------------------------------------------------------------------
load('W_paper_sim_true.mat');
%% Compute Correlation Matrix between W of different windows
%--------------------------------------------------------------------------
Nwindows = length(W_hat_vec);
W_corr_mean = zeros(Nwindows);
W_corr_mean_smooth = zeros(Nwindows);
W_corr_std = zeros(Nwindows);
tic;
count = 0;
Ncount = Nwindows^2;
for idx_row=1:Nwindows
    for idx_col=1:Nwindows
        HH = cell2mat(H_cell{idx_col});
        %HH = Hw{idx_col};
        prov = W_hat{idx_row}*HH;
        W_corr_mean(idx_row, idx_col) = norm(mean(prov,2).*[0.8; 1; 1]);
        W_corr_std(idx_row, idx_col) = norm(std(prov,1,2));
        disp((count/Ncount)*100);
        count = count + 1;
    end
    W_corr_mean_smooth(idx_row, :) = smooth(W_corr_mean(idx_row, :),20);
end
toc
%--------------------------------------------------------------------------

%% Display correlation Matrix
%--------------------------------------------------------------------------
%figure();
%imagesc(W_corr_mean); colorbar; colormap jet;
cenas = W_corr_mean./max(max(W_corr_mean));
figure();
imagesc(cenas); colorbar; colormap jet;
%caxis([0 0.000001]);
%title('normalized residual Wi * Hj');
xlabel('Number of the sub-dataset used to compute mean residual');
ylabel('Number of the sub-dataset used to estimate constraint parameters');

%caxis([0 0.01]);
%figure();
%imagesc(W_corr_std); colorbar; colormap jet;
%caxis([0 2]);
%figure();
%imagesc(W_corr_mean_smooth); colorbar; colormap jet;
%--------------------------------------------------------------------------