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
%file_name = getWFileName(window_size, increment);
%load(file_name);

%--------------------------------------------------------------------------

load('W_A_true_simdata.mat');

%% Compute Correlation Matrix between W of different windows
%--------------------------------------------------------------------------
Nwindows = length(W_hat_vec);
W_corr_mean = zeros(Nwindows);
W_corr_std = zeros(Nwindows);
tic;
for idx_row=1:Nwindows
    disp((idx_row/Nwindows)*100);
    wa = W_A_hat{idx_row};
%     wb = W_b_hat{idx_row};
    for idx_col=1:Nwindows
        A = cellfun(@(phia) wa*phia,...
            Phi_A_cell{idx_col},...
            'UniformOutput',false);
%         b = cellfun(@(phib) wb*phib,...
%             Phi_b_cell{idx_col},...
%             'UniformOutput',false);
        residual_cell = cellfun(@(A,u) pinv(A)*A*u,...
            A,u{idx_col},...
            'UniformOutput',false);
%         residual_cell = cellfun(@(A,u) pinv(A)*A*(u),...
%             A,u{idx_col},...
%             'UniformOutput',false);
        
        
        
%         residual_cell = cellfun(@(phia,h) pinv(wa*phia)*W_hat{idx_row}*h,...
%             Phi_A_cell{idx_col}, H_cell{idx_col},...
%             'UniformOutput',false);
        
        
        
        
        
%         residual_cell = cellfun(@(phi_inv,phiau) phi_inv*W_A_hat_prod{idx_row}*phiau, ...
%             Phi_A_inv_cell{idx_col}, Phi_Au_cell{idx_col}, ...
%             'UniformOutput',false);
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
%figure();
%imagesc(W_corr_std); colorbar; colormap jet;
%caxis([0 2]);
%--------------------------------------------------------------------------