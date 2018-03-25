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
% March 2018; Last revision: 17-Mar-2018

%% Add path
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
addpath(genpath('../')); % add the library and it's subfolders to the path
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Generate data
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Generating data ...\n');
c = {[0;0;0],[0;0;0]};
r = {1,1};
n = {[0; sind(-60); cosd(-60)], [0; sind(60); cosd(60)]};
NDem = numel(n);
freq = 100; % frequency of the data
tf = 5; % duration of the simulation
Nt = tf*freq; % number of training data points
theta = linspace(0,2*pi,Nt + 1);
% Initialize cells for storing data:
x = cell(1, NDem); % configurations
u = cell(1, NDem); % actions
t = cell(1, NDem); % configuration data
parfor idx=1:NDem
    % Null space base of the constraint:
    v=null(n{idx}.');
    % Data points:
    points=repmat(c{idx},1,size(theta,2))+r{idx}*(v(:,1)*cos(theta)+v(:,2)*sin(theta));
    % Store data points as cell:
    u{idx} = num2cell(freq.*diff(points,1,2),1); % save actions: cartesian velocities
    x{idx} = num2cell(points(:,1:end-1),1); % save state: cartesian positions
    t{idx} = num2cell(linspace(0,tf,Nt),1); % demonstration time
end
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Define constraint and main task regressors:
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Defining Phi_A ...\n');
Phi_A = @(q) eye(3); % Phi_A(x): vector of regressors for the Constraint matrix as a function of the configuration
PhiA_q = cell(1,NDem);
parfor idx=1:NDem
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
    [N_hat_phi{idx}, WA_hat{idx}] = ClosedForm_N_Estimatior(H_cell_phi{idx}, PhiA_q{idx}, constraint_dim);
end
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Compute unconstrained policy regressors:
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Computing Unconstrained Policy Regressors ...\n');
Phi = @(q) kron(q.', eye(3));
Phi_q = cell(1,NDem);
parfor idx=1:NDem
    Phi_q{idx} = cellfun(Phi, x{idx}, 'UniformOutput',false);
end
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Compute model variance
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Computing data variance...\n');
xall = cell2mat([x{:}]).';
scale = 0.2;
model_CAPL.var = scale.*std(xall,1,1).';
model_DPL.var = scale.*std(xall,1,1).';
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Compute model local Gaussian receptive fields centres'
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Computing Receptive Fields Centres ...\n');
stream = RandStream('mlfg6331_64');  % Random number stream for parallel computation
options = statset('Display','off','MaxIter',10000,'UseParallel',1,'UseSubstreams',1,'Streams',stream);
Nmodels = 20;
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
policy_CAPL = def_weighted_linear_model(model_CAPL, Phi);
policy_DPL = def_weighted_linear_model(model_DPL, Phi);
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Computing end-effector positions
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Computing End-Effector positions...\n');
traj_CAPL_dx = cell(1, NDem); % joints trajectory
traj_DPL_dx = cell(1, NDem); % joints trajectory
for idx=1:NDem
    % Problem specific constants taken from data:
    x0 = x{idx}{1}; % initial configuration
    % Constant matrices:
    W_A = n{idx}.'; % constant gain matrix for the Constraint matrix
    % Definition of Constraint matrix and main task
    A = @(x) W_A*feval(Phi_A,x); % Constraint matrix as a function of configuration
    b = @(x) 0;
    % Constrained Policie
    dx_CAPL = def_constrained_policy(A, b, policy_CAPL);
    dx_DPL = def_constrained_policy(A, b, policy_DPL);
    % solving motion
    sol_DPL_dx = ode113(@(t,x) dx_DPL(x),[0 1.566.*t{idx}{end}], x0);
    [traj_DPL_dx{idx}, ~] = deval(sol_DPL_dx,1.566.*cell2mat(t{idx})); % evaluation of solution
    sol_CAPL_dx = ode113(@(t,x) dx_CAPL(x),[0 t{idx}{end}], x0);
    [traj_CAPL_dx{idx}, ~] = deval(sol_CAPL_dx,cell2mat(t{idx})); % evaluation of solution
end
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Plot end-effector positions: training data
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Plotting Results...\n');
figure(); hold on;
kk = 5;
for idx=1:NDem
    % plot
    pos_data = downsample(cell2mat(x{idx})',kk)';
    pos_DPL = downsample(traj_DPL_dx{idx}',kk)';
    pos_CAPL = downsample(traj_CAPL_dx{idx}',kk)';
    plot3(pos_data(1,:),pos_data(2,:),pos_data(3,:),'g-+'); hold on;
    plot3(pos_DPL(1,:),pos_DPL(2,:),pos_DPL(3,:),'b-o');
    plot3(pos_CAPL(1,:),pos_CAPL(2,:),pos_CAPL(3,:),'r-*');
    plot3(x{idx}{1}(1),x{idx}{1}(2),x{idx}{1}(3),'xk','MarkerSize', 15,'linewidth',4);
end
grid on; axis image;
view(140,35);
legend('data','DPL','CAPL','x_0','Location','best');
xlabel('x [m]'); ylabel('y [m]'); zlabel('z [m]');
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Evaluate Constrained policies in grid:
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Evaluating Constrained policies in grid ...\n');
extract = @(C, k) cellfun(@(c)c(k), C);
%--------------------------------------------------------------------------
N = 9;
RR = linspace(-1,1,N); [XX,YY] = meshgrid(RR,RR); % state space
ZZ = zeros(size(XX));
% Definition of Constraint matrix and main task
A = @(x) [0 0 1]*feval(Phi_A,x); % Constraint matrix as a function of configuration
b = @(x) 0;
% Constrained Policie
dx_DPL = def_constrained_policy(A, b, policy_DPL);
dx_CAPL = def_constrained_policy(A, b, policy_CAPL);
% evaluation:
out_DPL = arrayfun(@(x1,x2,x3) dx_DPL([x1; x2; x3]), XX, YY, ZZ,'UniformOutput', false);
out_CAPL = arrayfun(@(x1,x2,x3) dx_CAPL([x1; x2;x3]), XX, YY, ZZ,'UniformOutput', false);
% Extract actions
UU_DPL= extract(out_DPL, 1); VV_DPL= extract(out_DPL, 2);
UU_CAPL= extract(out_CAPL, 1); VV_CAPL= extract(out_CAPL, 2);
% Plots
figure(); hold on;
quiver(XX,YY,UU_DPL,VV_DPL,'b');
quiver(XX,YY,UU_CAPL,VV_CAPL,'r');
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Compute end-effector positions
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Computing End-Effector positions...\n');
% Initial configuration:
x0 = [1; 0; 0]; % initial configuration
% solving motion
sol_DPL_dx = ode113(@(t,x) dx_DPL(x),[0 t{1}{end}], x0);
[traj_DPL_dx, ~] = deval(sol_DPL_dx,cell2mat(t{1})); % evaluation of solution
sol_CAPL_dx = ode113(@(t,x) dx_CAPL(x),[0 0.5.*t{1}{end}], x0);
[traj_CAPL_dx, ~] = deval(sol_CAPL_dx,0.5.*cell2mat(t{1})); % evaluation of solution
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Plot end-effector positions: training data
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Plotting Results...\n');
kk = 5;
pos_DPL = downsample(traj_DPL_dx',kk)';
pos_CAPL = downsample(traj_CAPL_dx',kk)';
plot(pos_DPL(1,:),pos_DPL(2,:),'b-o');
plot(pos_CAPL(1,:),pos_CAPL(2,:),'r-*');
plot(x0(1),x0(2),'xk','MarkerSize', 15,'linewidth',4);
axis image; grid on;
xlabel('x [m]'); ylabel('y [m]');
legend('DPL','CAPL','DPL traj.','CAPL traj.','x_0','Location','best');
hold off;
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
