function [W_hat, W_hat_vec, Hi] = estimateW4windows(H, ConstraintDim, increment, window_size)
Nsamples = length(H);
Nwindows = floor((Nsamples-window_size)/increment)+1;
W_hat = cell(1,Nwindows);
W_hat_vec = cell(1,Nwindows);
Hi = cell(1,Nwindows);
first = (0:1:(Nwindows-1)).*increment+1;
last = first + window_size - 1;
for idx=1:Nwindows
    Hi{idx} = H(:,first(idx):last(idx));
    [Ui,~,~]=svd(Hi{idx});
    W_hat{idx} = Ui(:,(end-(ConstraintDim-1)):end).';
    W_hat_vec{idx} = reshape(W_hat{idx},[],1);
end
end
