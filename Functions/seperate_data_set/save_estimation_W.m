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
load('data_smooth.mat');
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
Phi_b = def_phib_4_spm_sim(robot); % Phi_b(q): vector of regressors for the main task as a function of the configuration
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
file_name = 'H.mat';
if exist(file_name,'file')
    load(file_name);
else
    H_cell = cell(size(x_all));
    parfor idx=1:length(H_cell)
        % H(q,u): Matrix of regressors as a function of the configuration and action compute number of regressors
        H_cell{idx} = [feval(Phi_A, x_all{idx})*u_all{idx}; -feval(Phi_b,x_all{idx})];
    end
    H = cell2mat(H_cell);
    save(file_name,'H','H_cell');
end
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%% Estimating weights with the all data sets together and saving to file
fprintf(1, 'Estimating W ...\n');
getWFileName = @(w,i) strcat('W/w',num2str(w),'i',num2str(i),'.mat');
[W_hat, W_hat_vec, Hw] = estimateW4windows(H, 3, 100, 1000);
% tic;
% ConstraintDim = 3;
% for window_size = 400:100:2500
%     for increment = 50:50:200
%         [W_hat, W_hat_vec, Hw] = estimateW4windows(H, ConstraintDim, increment, window_size);
%         save(getWFileName(window_size, increment),'W_hat','W_hat_vec','Hw');
%         clear W_hat W_hat_vec Hw;
%     end
% end
% toc
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------


%% Remove path
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
rmpath(genpath('../'));
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------