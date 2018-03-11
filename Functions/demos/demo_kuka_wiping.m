% Given the data composed of states - joint positions - and actions - joint
% velocities, we estimate the null space projection matrix for each data 
% set/demonstration and use that result to compute the unconstrained policy.
% We then plot the result of the policy and estimated projection matrix with 
% the input data for the kuka end-effector cartesian positions.
%
% Other m-files required: 
%   def_phi_4_cwm.m
%   def_phib_4_spm_sim.m
%   def_phib_4_spm_exp.m
%   def_phia_4_spm.m
%   def_constrained_policy.m

% Author: Joao Moura
% Edinburgh Centre for Robotics, Edinburgh, UK
% email address: Joao.Moura@ed.ac.uk
% Website: http://www.edinburgh-robotics.org/students/joao-moura
% October 2017; Last revision: 09-Mar-2018

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

%% Initialize roobot model
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
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Get data
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Getting data ...\n');
file_name = 'data_smooth';
file_ext = '.mat';
load(strcat(file_name,file_ext));
NDem = length(x); % number of demonstrations
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Compute end-effector position:
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Computing end-effector position ...\n');
getPos = @(q) transl(robot.fkine(q)); % compute end-effector postion
file_pos = strcat(file_name,'_pos',file_ext);
if exist(file_pos, 'file') == 2
    load(file_pos);
else
    p = cell(1, NDem); % end-effector cartesian position in global frame
    tic;
    parfor idx=1:NDem
        p{idx} = getPos(cell2mat(x{idx}).'); % compute end-effector postion
    end
    toc
    save(file_pos);
end
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Define constraint and main task regressors:
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Defining Phi_A and Phi_b ...\n');
Phi_A = def_phia_4_spm(robot); % Phi_A(x): vector of regressors for the Constraint matrix as a function of the configuration
Phi_b = def_phib_4_spm_sim(robot); % Phi_b(x): vector of regressors for the main task as a function of the configuration
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Evaluate constraint and main task regressors:
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Evaluating Phi_A and Phi_b ...\n');
file_phiAb = strcat(file_name,'_phiAq',file_ext);
if exist(file_phiAb, 'file') == 2
    load(file_phiAb);
else
    PhiA_q = cell(1,NDem);
    Phib_q = cell(1,NDem);
    tic;
    parfor idx=1:NDem
        PhiA_q{idx} = cellfun(Phi_A, x{idx}, 'UniformOutput',false);
        Phib_q{idx} = cellfun(Phi_b, x{idx}, 'UniformOutput',false);
    end
    toc
    save(file_phiAb);
end
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Estimate the null space projection matrix for each demonstration
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Estimating constraints ...\n');
H = @(phiAq,phibq,u) phiAq*u;
%H = @(phiAq,phibq,u) [phiAq*u; -phibq];
constraint_dim = 3;
% Initialize cell variables:
N_hat_phi = cell(1,NDem);
H_cell_phi = cell(1,NDem);
% Compute null space projection matrix estimation
parfor idx=1:NDem
    % Compute H regressors
    H_cell_phi{idx} = cellfun(H, PhiA_q{idx}, Phib_q{idx}, u{idx}, 'UniformOutput',false);
    % Estimate weights for the null space projection matrix
    N_hat_phi{idx} = ClosedForm_N_Estimatior(H_cell_phi{idx}, PhiA_q{idx}, constraint_dim)
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
Phi = cell(1,NDem);
parfor idx=1:NDem
    [c{idx}, r{idx}, n{idx}] = fit_3d_circle(p{idx}(:,1),p{idx}(:,2),p{idx}(:,3));
    Phi{idx} = def_phi_4_cwm_sim(robot, c{idx}, r{idx}); % Get regressors for the unconstrained policy
end

%% Compute unconstrained policy regressors:
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Computing Unconstrained Policy Regressors ...\n');
file_phi = strcat(file_name,'_phi',file_ext);
if exist(file_phi, 'file') == 2
    load(file_phi);
else
    Phi_q = cell(1,NDem);
    tic;
    parfor idx=1:NDem
        Phi_q{idx} = cellfun(Phi{idx}, x{idx}, 'UniformOutput',false);
    end
    toc
    save(file_phi);
end
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Compute model variance
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Computing data variance...\n');
xall = cell2mat([x{:}]).';
scale = 2;
model.var = scale.*std(xall,1,1).';
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Compute model local Gaussian receptive fields centres'
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Computing Receptive Fields Centres ...\n');
stream = RandStream('mlfg6331_64');  % Random number stream for parallel computation
options = statset('Display','off','MaxIter',200,'UseParallel',1,'UseSubstreams',1,'Streams',stream);
Nmodels = 25;
[~,C] = kmeans(xall,Nmodels,'Distance','cityblock','EmptyAction','singleton','Start','uniform',...
    'Replicates',10,'OnlinePhase','off','Options', options);
model.c = C.';
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
R = cell2mat([R_cell{:}].');
% Compute null space component of actions: u_ns
u_ns = cell(1,NDem);
parfor idx=1:NDem
    u_ns{idx} = cellfun(@(Nhat,u) Nhat*u, N_hat_phi{idx}, u{idx}, 'UniformOutput',false);
end
Y = cell2mat([u_ns{:}].');
% Weighting matrix:
B = zeros(size(Phi{1}(x{1}{1}),2),size(model.c,2));
w = @(m) @(x) exp(-0.5.*sum(bsxfun(@rdivide, bsxfun(@minus,x,model.c(:,m)).^2, model.var))).'; % importance weights W = [w1 w2 ... w_m ... w_M]
[nRrow,nRcol] = size(R_cell{1}{1});
parfor m=1:size(model.c,2)
    wm = feval(w, m);
    Wm = repelem(wm(xall.'),nRrow,nRcol);
    RWm = R.*Wm;
    B(:,m) = pinv(RWm.'*R)*RWm.'*Y;
end
model.b = B;
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Defining Unconstrained Policies
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Computing Unconstrained Policy...\n');
policy_CAL = cell(1,NDem);
parfor idx=1:NDem
    policy_CAL{idx} = def_weighted_linear_model(model, Phi{idx});
end
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Computing end-effector positions based on learned policies
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Compute End-Effector positions...\n');
plot_idx = [1,2,4,7];
Nidx = numel(plot_idx);
pos = cell(1, Nidx); % wiping circle centre
traj = cell(1, Nidx); % joints trajectory
% Select subset of trajectories
x_ = x(plot_idx);
t_ = t(plot_idx);
n_ = n(plot_idx);
c_ = c(plot_idx);
policy_CAL_ = policy_CAL(plot_idx);
parfor idx=1:Nidx
    % Problem specific constants taken from data:
    x0 = x_{idx}{1}; % initial configuration
    Kp = 1; % proportional gain
    % Constant matrices:
    W_A = blkdiag(n_{idx}.', n_{idx}.', n_{idx}.'); % constant gain matrix for the Constraint matrix
    W_b = -Kp*[W_A [-n_{idx}.'*c_{idx}; 0; 0]];
    % Definition of Constraint matrix and main task
    A = @(x) W_A*feval(Phi_A,x); % Constraint matrix as a function of configuration
    b = @(x) W_b*feval(Phi_b,x); % main task as a function of the configuration
    % Constrained Policie
    dx = def_constrained_policy(A, b, policy_CAL_{idx});
    % solving motion
    sol = ode113(@(t,x) dx(x),[0 t_{idx}{end}], x0);
    [traj_{idx}, ~] = deval(sol,cell2mat(t_{idx})); % evaluation of solution
    %pos=transl(robot.fkine(traj));
    pos_{idx}=getPos(traj_{idx}.');
end
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Plot end-effector positions
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Plotting Results...\n');
figure(); hold on;
p_ = p(plot_idx);
%for idx=1:NDem
for idx=1:Nidx
    % plot
    plot3(p_{idx}(:,1),p_{idx}(:,2),p_{idx}(:,3),'g');
    plot3(pos_{idx}(:,1),pos_{idx}(:,2),pos_{idx}(:,3),'b');
end
legend('data','policy','Location','best');
xlabel('x [m]'); ylabel('y [m]'); zlabel('z [m]');
grid on;
axis equal;
view(-60,20);
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
