%% User input
%--------------------------------------------------------------------------
W_file = 'W_A_new';
window_size = 1200;
increment = 100;
%--------------------------------------------------------------------------

%% Auxiliar functions:
%--------------------------------------------------------------------------
getWFileName = @(w,i) strcat(W_file,'/w',num2str(w),'i',num2str(i),'.mat');
%--------------------------------------------------------------------------

%% Load estimated gains W
%--------------------------------------------------------------------------
file_name = getWFileName(window_size, increment);
load(file_name);
%--------------------------------------------------------------------------

%load('W_true_A.mat');
%% Compute Correlation Matrix between W of different windows
%--------------------------------------------------------------------------
Nwindows = length(W_hat_vec);
W_corr_mean = zeros(Nwindows);
W_corr_std = zeros(Nwindows);
tic;
for idx_row=1:Nwindows
    disp((idx_row/Nwindows)*100);
    W_hat_prod_const = W_hat_prod{idx_row};
    for idx_col=1:Nwindows
        residual_cell = cellfun(@(phi_inv,h) phi_inv*W_hat_prod_const*h, ...
            Phi_A_inv_cell_w{idx_col}, Hw{idx_col}, ...
            'UniformOutput',false);
        residual = cell2mat(residual_cell);
        W_corr_mean(idx_row, idx_col) = norm(mean(residual,2));
        W_corr_std(idx_row, idx_col) = norm(std(residual,1,2));
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
%caxis([0 2]);
%--------------------------------------------------------------------------