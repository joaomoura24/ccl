%% Add path
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
addpath(genpath('../'));
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Get data
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Getting data ...\n');
load('data_paper_sim.mat');
NDem = length(x); % number of demonstrations
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Initialize roobot model and the Regressors for the constraint and main task
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Defining robot model ...\n');
DH = [0.0, 0.31, 0.0, pi/2; % Robot Kinematic model specified by the Denavit-Hartnbergh
      0.0, 0.0, 0.0, -pi/2;
      0.0, 0.4, 0.0, -pi/2;
      0.0, 0.0, 0.0, pi/2;
      0.0, 0.39, 0.0, pi/2;
      0.0, 0.0, 0.0, -pi/2;
      0.0, 0.21-0.132, 0.0, 0.0];
robot = SerialLink(DH); % Peters Cork robotics library has to be installed
Phi_A = def_phia_4_spm(robot); % Phi_A(q): vector of regressors for the Constraint matrix as a function of the configuration
Phi_b = def_phib_4_spm_sim(robot);
file_name = 'paper_sim_true';
H_fun = @(phiau,phib) [phiau; -phib];
%H_fun = @(phiau,phib) phiau;
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Parallel computig settingsfeval(
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Initializing parallel pool ...\n');
gcp(); % Get the current parallel pool
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Computing H matrix
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Computing H ...\n');
file_name_H = strcat('H_',file_name,'.mat');
if exist(file_name_H,'file')
    load(file_name_H);
else
    H_cell = cell(1, NDem);
    Phi_A_cell = cell(1,NDem);
    Phi_b_cell = cell(1,NDem);
    Phi_Au_cell = cell(1,NDem);
    Phi_A_inv_cell = cell(1,NDem);
    parfor idx=1:NDem
        Phi_A_cell{idx} = cellfun(Phi_A, x{idx},...
            'UniformOutput',false);
        Phi_b_cell{idx} = cellfun(Phi_b, x{idx},...
            'UniformOutput',false);
        Phi_Au_cell{idx} = cellfun(@(phia,u) phia*u,...
            Phi_A_cell{idx}, u{idx},'UniformOutput',false);
        Phi_A_inv_cell{idx} = cellfun(@(f) pinv(f), Phi_A_cell{idx},...
            'UniformOutput',false);
        % H(q,u): Matrix of regressors as a function of the configuration and action compute number of regressors
        H_cell{idx} = cellfun(H_fun, Phi_Au_cell{idx},...
            Phi_b_cell{idx},'UniformOutput',false);
    end
    save(file_name_H,...
        'H_cell','Phi_A_cell','Phi_b_cell','Phi_A_inv_cell','Phi_Au_cell','u');
end
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------


%% Estimating weights with the all data sets together and saving to file
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1, 'Estimating W ...\n');
tic;
ConstraintDim = 3;
basisDim = length(Phi_Au_cell{1}{1});
W_hat = cell(1,NDem);
W_A_hat = cell(1,NDem);
W_b_hat = cell(1,NDem);
W_A_hat_prod = cell(1,NDem);
W_hat_vec = cell(1,NDem);
for idx=1:NDem
    [Ui,Si,~]=svd(cell2mat(H_cell{idx}));
    kk = 0.1;
    disp(sum(diag(Si)<kk));
    %ConstraintDim = sum(diag(Si)<kk);
    W_hat{idx} = Ui(:,(end-(ConstraintDim-1)):end).';
    W_A_hat{idx} = W_hat{idx}(:,1:basisDim);
    W_b_hat{idx} = W_hat{idx}(:,(basisDim+1):end);
    W_A_hat_prod{idx} = pinv(W_A_hat{idx})*W_A_hat{idx};
    W_hat_vec{idx} = reshape(W_hat{idx},[],1);
end
file_name_W = strcat('W_',file_name,'.mat');
save(file_name_W,...
    'W_hat','W_A_hat','W_b_hat','W_A_hat_prod','W_hat_vec','H_cell','Phi_A_cell','Phi_b_cell','Phi_A_inv_cell','Phi_Au_cell','u');
toc
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------


%% Remove path
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
rmpath(genpath('../'));
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------