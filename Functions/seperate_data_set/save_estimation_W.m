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
load('data_test_paper.mat');
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
x_all = [x{:}]; % all states together as cell
u_all = [u{:}]; % all actions together as cell
file_name = 'H_test_paper.mat';
H_fun = @(phia,phib,u) [phia*u; -phib];
if exist(file_name,'file')
    load(file_name);
else
    H_all = cell(size(x_all));
    Phi_A_cell = cell(size(x_all));
    Phi_b_cell = cell(size(x_all));
    parfor idx=1:length(H_all)
        % H(phi,u): Matrix of regressors as a function of the configuration and action compute number of regressors
        Phi_A_cell{idx} = feval(Phi_A, x_all{idx});
        Phi_b_cell{idx} = feval(Phi_b, x_all{idx});
        H_all{idx} = feval(H_fun, Phi_A_cell{idx},...
            Phi_b_cell{idx},u_all{idx});
    end
    save(file_name,'H_all','x_all','u_all','Phi_A_cell','Phi_b_cell');
end
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------


%% Estimating weights with the all data sets together and saving to file
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1, 'Estimating W ...\n');
getWFileName = @(w,i) strcat('W_test/w',num2str(w),'i',num2str(i),'.mat');
tic;
ConstraintDim = 3;
for window_size = 400
    for increment = 50
        [W_hat, W_hat_vec, H_cell] = estimateW4windows(...
            H_all, ConstraintDim, increment, window_size);
        save(getWFileName(window_size, increment),...
            'W_hat','W_hat_vec','H_cell');
        clear W_hat W_hat_vec H_cell;
    end
end
toc
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------


%% Remove path
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
rmpath(genpath('../'));
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------