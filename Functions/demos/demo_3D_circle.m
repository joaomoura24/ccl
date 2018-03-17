% Given the data composed of states - Cartesian positions - and actions -
% Cartesian velocities, we estimate the null space projection matrix for each data 
% set/demonstration and use that result to compute the unconstrained policy.
% We then plot the result of the policy and estimated projection matrix with 
% the input data.
%
% Author: Joao Moura
% Edinburgh Centre for Robotics, Edinburgh, UK
% email address: Joao.Moura@ed.ac.uk
% Website: http://www.edinburgh-robotics.org/students/joao-moura
% March 2018; Last revision: 15-Mar-2018

%% Add path
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
addpath(genpath('../')); % add the library and it's subfolders to the path
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Parallel computig settingsfeval(
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Initializing parallel pool ...\n');
gcp(); % Get the current parallel pool
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Get data
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Getting data ...\n');
file_name = 'data_3D_circles_training';
file_ext = '.mat';
load(strcat(file_name,file_ext));
NDem = length(x); % number of demonstrations
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Define constraint and main task regressors:
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Defining Phi_A ...\n');
Phi_A = @(q) eye(3); % Phi_A(x): vector of regressors for the Constraint matrix as a function of the configuration
PhiA_q = cell(1,NDem);
for idx=1:NDem
    PhiA_q{idx} = cellfun(Phi_A, x{idx}, 'UniformOutput',false);
end
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Estimate the null space projection matrix for each demonstration
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Estimating constraints ...\n');
H = @(phia,u) phia*u;
constraint_dim = 1;
% Initialize cell variables:
N_hat_phi = cell(1,NDem);
H_cell_phi = cell(1,NDem);
WA_hat = cell(1,NDem);
% Compute null space projection matrix estimation
parfor idx=1:NDem
    % Compute H regressors
    H_cell_phi{idx} = cellfun(H, PhiA_q{idx}, u{idx}, 'UniformOutput',false);
    % Estimate weights for the null space projection matrix
    [N_hat_phi{idx}, WA_hat{idx}] = ClosedForm_N_Estimatior(H_cell_phi{idx}, PhiA_q{idx}, constraint_dim)
end
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Define Policy Regressors for each demonstration
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Defining Unconstrained Policy Regressors ...\n');
c = cell(1, NDem); % wiping circle centre
r = cell(1, NDem); % wiping circle radious
n = cell(1, NDem); % planar surface normal
pos = cell(1, NDem); % position
Phi = cell(1,NDem);
for idx=1:NDem
    pos{idx} = cell2mat(x{idx});
    [c{idx}, r{idx}, n{idx}] = fit_3d_circle(pos{idx}(1,:).',pos{idx}(2,:).',pos{idx}(3,:).');
    Phi{idx} = def_phi_circ();
end
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Compute unconstrained policy regressors:
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Computing Unconstrained Policy Regressors ...\n');
parfor idx=1:NDem
    Phi_q{idx} = cellfun(Phi{idx}, x{idx}, 'UniformOutput',false);
end
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Compute model variance
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Computing data variance...\n');
xall = cell2mat([x{:}]).';
scale = 1;
model_CAPL.var = scale.*std(xall,1,1).';
model_DPL.var = scale.*std(xall,1,1).';
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Compute model local Gaussian receptive fields centres'
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Computing Receptive Fields Centres ...\n');
stream = RandStream('mlfg6331_64');  % Random number stream for parallel computation
options = statset('Display','off','MaxIter',1000,'UseParallel',1,'UseSubstreams',1,'Streams',stream);
Nmodels = 1;
[~,C] = kmeans(xall,Nmodels,'Distance','cityblock','EmptyAction','singleton','Start','uniform',...
    'Replicates',10,'OnlinePhase','off','Options', options);
model_CAPL.c = C.';
model_DPL.c = C.';
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Compute model parameters
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Learning Model Parameters...\n');
R_cell = cell(1,NDem);
% Compute matrix R: policy regressors times estimated null space prjection:
parfor idx=1:NDem
    R_cell{idx} = cellfun(@(Nhat,phi) Nhat*phi, N_hat_phi{idx}, Phi_q{idx}, 'UniformOutput',false);
end
R_CAPL = cell2mat([R_cell{:}].');
R_DPL = cell2mat([Phi_q{:}].');
% Compute null space component of actions: u_ns
u_ns = cell(1,NDem);
parfor idx=1:NDem
    u_ns{idx} = cellfun(@(Nhat,u) Nhat*u, N_hat_phi{idx}, u{idx}, 'UniformOutput',false);
end
Y_CAPL = cell2mat([u_ns{:}].');
Y_DPL = cell2mat([u{:}].');
% Weighting matrix:
B_CAPL = zeros(size(Phi_q{1}{1},2),size(model_CAPL.c,2));
B_DPL = zeros(size(Phi_q{1}{1},2),size(model_DPL.c,2));
w = @(m) @(x) exp(-0.5.*sum(bsxfun(@rdivide, bsxfun(@minus,x,model_CAPL.c(:,m)).^2, model_CAPL.var))).'; % importance weights W = [w1 w2 ... w_m ... w_M]
[nRrow,nRcol] = size(R_cell{1}{1});
parfor m=1:size(model_CAPL.c,2)
    wm = feval(w, m);
    Wm = repelem(wm(xall.'),nRrow,nRcol);
    RWm_CAPL = R_CAPL.*Wm;
    RWm_DPL = R_DPL.*Wm;
    B_CAPL(:,m) = pinv(RWm_CAPL.'*R_CAPL)*RWm_CAPL.'*Y_CAPL;
    B_DPL(:,m) = pinv(RWm_DPL.'*R_DPL)*RWm_DPL.'*Y_DPL;
end
model_CAPL.b = B_CAPL;
model_DPL.b = B_DPL;
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Defining Unconstrained Policies
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Computing Unconstrained Policy...\n');
policy_CAPL = cell(1,NDem);
policy_DPL = cell(1,NDem);
parfor idx=1:NDem
    policy_CAPL{idx} = def_weighted_linear_model(model_CAPL, Phi{idx});
    policy_DPL{idx} = def_weighted_linear_model(model_DPL, Phi{idx});
end
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Computing end-effector positions based on learned policies: training data
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Compute End-Effector positions...\n');
traj_CAPL_dx = cell(1, NDem); % joints trajectory
traj_DPL_dx = cell(1, NDem); % joints trajectory
traj_CAPL_pi = cell(1, NDem); % joints trajectory
traj_DPL_pi = cell(1, NDem); % joints trajectory
for idx=1:NDem
    % Problem specific constants taken from data:
    x0 = x{idx}{1}; % initial configuration
    % Constant matrices:
    W_A = n{idx}.'; % constant gain matrix for the Constraint matrix
    % Definition of Constraint matrix and main task
    A = @(x) W_A*feval(Phi_A,x); % Constraint matrix as a function of configuration
    b = @(x) 0;
    % Constrained Policie
    dx_CAPL = def_constrained_policy(A, b, policy_CAPL{idx});
    dx_DPL = def_constrained_policy(A, b, policy_DPL{idx});
    % solving motion
    sol_CAPL_dx = ode113(@(t,x) dx_CAPL(x),[0 t{idx}{end}], x0);
    [traj_CAPL_dx{idx}, ~] = deval(sol_CAPL_dx,cell2mat(t{idx})); % evaluation of solution
    sol_DPL_dx = ode113(@(t,x) dx_DPL(x),[0 t{idx}{end}], x0);
    [traj_DPL_dx{idx}, ~] = deval(sol_DPL_dx,cell2mat(t{idx})); % evaluation of solution
    sol_CAPL_pi = ode113(@(t,x) policy_CAPL{idx}(x),[0 t{idx}{end}], x0);
	[traj_CAPL_pi{idx}, ~] = deval(sol_CAPL_pi,cell2mat(t{idx})); % evaluation of solution
	sol_DPL_pi = ode113(@(t,x) policy_DPL{idx}(x),[0 t{idx}{end}], x0);
	[traj_DPL_pi{idx}, ~] = deval(sol_DPL_pi,cell2mat(t{idx})); % evaluation of solution
end
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Plot end-effector positions: training data
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Plotting Results...\n');
figure();
for idx=1:NDem
    subplot(2,3,idx);
    % plot
    plot3(pos{idx}(1,:),pos{idx}(2,:),pos{idx}(3,:),'g'); hold on;
    plot3(traj_CAPL_pi{idx}(1,:),traj_CAPL_pi{idx}(2,:),traj_CAPL_pi{idx}(3,:),'b');
    plot3(traj_DPL_pi{idx}(1,:),traj_DPL_pi{idx}(2,:),traj_DPL_pi{idx}(3,:),'r');
    grid on; axis image;
    legend('data','CAPL','DPL','Location','best');
    xlabel('x [m]'); ylabel('y [m]'); zlabel('z [m]');
end
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Plot end-effector positions: training data
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Plotting Results...\n');
figure();
for idx=1:NDem
    subplot(2,3,idx);
    % plot
    plot3(pos{idx}(1,:),pos{idx}(2,:),pos{idx}(3,:),'g'); hold on;
    plot3(traj_CAPL_dx{idx}(1,:),traj_CAPL_dx{idx}(2,:),traj_CAPL_dx{idx}(3,:),'b');
    plot3(traj_DPL_dx{idx}(1,:),traj_DPL_dx{idx}(2,:),traj_DPL_dx{idx}(3,:),'r');
    grid on; axis image;
    legend('data','CAPL','DPL','Location','best');
    xlabel('x [m]'); ylabel('y [m]'); zlabel('z [m]');
end

%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Get data: test
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Getting data ...\n');
file_name = 'data_3D_circles_test';
file_ext = '.mat';
load(strcat(file_name,file_ext));
NDem = length(x); % number of demonstrations
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Define Policy Regressors for each demonstration
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Defining Unconstrained Policy Regressors ...\n');
c = cell(1, NDem); % wiping circle centre
r = cell(1, NDem); % wiping circle radious
n = cell(1, NDem); % planar surface normal
pos = cell(1, NDem); % position
Phi = cell(1,NDem);
for idx=1:NDem
    pos{idx} = cell2mat(x{idx});
    [c{idx}, r{idx}, n{idx}] = fit_3d_circle(pos{idx}(1,:).',pos{idx}(2,:).',pos{idx}(3,:).');
    Phi{idx} = def_phi_circ();
end
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Computing end-effector positions based on learned policies: training data
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Compute End-Effector positions...\n');
traj_CAPL_dx = cell(1, NDem); % joints trajectory
traj_DPL_dx = cell(1, NDem); % joints trajectory
traj_CAPL_pi = cell(1, NDem); % joints trajectory
traj_DPL_pi = cell(1, NDem); % joints trajectory
for idx=1:NDem
    % Problem specific constants taken from data:
    x0 = x{idx}{1}; % initial configuration
    % Constant matrices:
    W_A = n{idx}.'; % constant gain matrix for the Constraint matrix
    % Definition of Constraint matrix and main task
    A = @(x) W_A*feval(Phi_A,x); % Constraint matrix as a function of configuration
    b = @(x) 0;
    % Constrained Policie
    dx_CAPL = def_constrained_policy(A, b, policy_CAPL{idx});
    dx_DPL = def_constrained_policy(A, b, policy_DPL{idx});
    % solving motion
    sol_CAPL_dx = ode113(@(t,x) dx_CAPL(x),[0 t{idx}{end}], x0);
    [traj_CAPL_dx{idx}, ~] = deval(sol_CAPL_dx,cell2mat(t{idx})); % evaluation of solution
    sol_DPL_dx = ode113(@(t,x) dx_DPL(x),[0 t{idx}{end}], x0);
    [traj_DPL_dx{idx}, ~] = deval(sol_DPL_dx,cell2mat(t{idx})); % evaluation of solution
    sol_CAPL_pi = ode113(@(t,x) policy_CAPL{idx}(x),[0 t{idx}{end}], x0);
	[traj_CAPL_pi{idx}, ~] = deval(sol_CAPL_pi,cell2mat(t{idx})); % evaluation of solution
	sol_DPL_pi = ode113(@(t,x) policy_DPL{idx}(x),[0 t{idx}{end}], x0);
	[traj_DPL_pi{idx}, ~] = deval(sol_DPL_pi,cell2mat(t{idx})); % evaluation of solution
end
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Plot end-effector positions: training data
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Plotting Results...\n');
figure();
for idx=1:NDem
    subplot(1,3,idx);
    % plot
    plot3(pos{idx}(1,:),pos{idx}(2,:),pos{idx}(3,:),'g'); hold on;
    plot3(traj_CAPL_pi{idx}(1,:),traj_CAPL_pi{idx}(2,:),traj_CAPL_pi{idx}(3,:),'b');
    plot3(traj_DPL_pi{idx}(1,:),traj_DPL_pi{idx}(2,:),traj_DPL_pi{idx}(3,:),'r');
    grid on; axis image;
    legend('data','CAPL','DPL','Location','best');
    xlabel('x [m]'); ylabel('y [m]'); zlabel('z [m]');
end
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Plot end-effector positions: training data
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Plotting Results...\n');
figure();
for idx=1:NDem
    subplot(1,3,idx);
    % plot
    plot3(pos{idx}(1,:),pos{idx}(2,:),pos{idx}(3,:),'g'); hold on;
    plot3(traj_CAPL_dx{idx}(1,:),traj_CAPL_dx{idx}(2,:),traj_CAPL_dx{idx}(3,:),'b');
    plot3(traj_DPL_dx{idx}(1,:),traj_DPL_dx{idx}(2,:),traj_DPL_dx{idx}(3,:),'r');
    grid on; axis image;
    legend('data','CAPL','DPL','Location','best');
    xlabel('x [m]'); ylabel('y [m]');
end
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Auxiliar functions
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Deleting parallel pool...\n');
%delete(gcp('nocreate'));
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Remove path
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
rmpath(genpath('../'));
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
