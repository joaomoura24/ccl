function [W_hat, W_hat_vec, Hi_cell] = estimateW4windows(H_cell, ConstraintDim, increment, window_size)
Nsamples = length(H_cell);
Nwindows = floor((Nsamples-window_size)/increment)+1;
W_hat = cell(1,Nwindows);
W_hat_vec = cell(1,Nwindows);
Hi_cell = cell(1,Nwindows);
first = (0:1:(Nwindows-1)).*increment+1;
last = first + window_size - 1;
for idx=1:Nwindows
    Hi_cell{idx} = H_cell(first(idx):last(idx));
    Hi = cell2mat(Hi_cell{idx});
    [Ui,~,~]=svd(Hi);
    W_hat{idx} = Ui(:,(end-(ConstraintDim-1)):end).';
    W_hat_vec{idx} = reshape(W_hat{idx},[],1);
end
end
