function [N_hat_phi, WA_hat] = ClosedForm_N_Estimatior(H_q, PhiA_q, ConstraintDim)
%------------- BEGIN CODE --------------
    H_matrix = cell2mat(H_q);
    % Singular value decomposition to estimate the gains
    [U,~,~]=svd(H_matrix);
    W_hat = U(:,(end-(ConstraintDim-1)):end).'; % Select the last ConstraintDim columns.
    % Select first NA columns in case of using some constrained action:
    NA = size(PhiA_q{1},1); % Number of the regressors for the constraint matrix.
    WA_hat = W_hat(:,1:NA);
    % Definition of Constraint matrix and null space projection:
    A_hat = @(phiAq) WA_hat*phiAq; % Constraint matrix as a function of configuration.
    N_hat = def_null_space_proj_phi(A_hat);
    N_hat_phi = cellfun(N_hat, PhiA_q, 'UniformOutput',false);
%------------- END OF CODE --------------
end
