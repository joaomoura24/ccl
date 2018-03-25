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
file_name = 'data_ref_sep2';
file_ext = '.mat';
load(strcat(file_name,file_ext));
window_test = 5;
x_test = x(window_test);
u_test = u(window_test);
t_test = t(window_test);
window_train = [1 2 4 6:10];
x = x(window_train);
u = u(window_train);
t = t(window_train);
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
    %Phi{idx} = def_phi_4_cwm_sim(robot, c{idx}, r{idx}); % Get regressors for the unconstrained policy
    Phi{idx} = def_phi_test(robot, c{idx});
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
options = statset('Display','off','MaxIter',200,'UseParallel',1,'UseSubstreams',1,'Streams',stream);
Nmodels = 100;
[~,C] = kmeans(xall,Nmodels,'Distance','cityblock','EmptyAction','singleton','Start','uniform',...
    'MaxIter',1000,'Replicates',10,'OnlinePhase','off','Options', options);
model_CAPL.c = C.';
model_DPL.c = C.';
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

%% Computing end-effector positions based on learned policies
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Compute End-Effector positions...\n');
%Nidx = 12; if NDem<Nidx; Nidx = NDem; end;
Nidx = NDem;
pos_CAPL_dx = cell(1, Nidx); % wiping circle centre
traj_CAPL_dx = cell(1, Nidx); % joints trajectory
pos_DPL_dx = cell(1, Nidx); % wiping circle centre
traj_DPL_dx = cell(1, Nidx); % joints trajectory
pos_CAPL_pi = cell(1, Nidx); % wiping circle centre
traj_CAPL_pi = cell(1, Nidx); % joints trajectory
pos_DPL_pi = cell(1, Nidx); % wiping circle centre
traj_DPL_pi = cell(1, Nidx); % joints trajectory
parfor idx=1:Nidx
    %time_end= 1;
    %time = linspace(0,time_end,100);
    % Problem specific constants taken from data:
    x0 = x{idx}{1}; % initial configuration
    Kp = 0.1; % proportional gain
    % Constant matrices:
    W_A = blkdiag(n{idx}.', n{idx}.', n{idx}.'); % constant gain matrix for the Constraint matrix
    W_b = -Kp*[W_A [-n{idx}.'*c{idx}; 0; 0]];
    % Definition of Constraint matrix and main task
    A = @(x) W_A*feval(Phi_A,x); % Constraint matrix as a function of configuration
    b = @(x) W_b*feval(Phi_b,x); % main task as a function of the configuration
    % Constrained Policie
    dx_CAPL = def_constrained_policy(A, b, policy_CAPL{idx});
    dx_DPL = def_constrained_policy(A, b, policy_DPL{idx});
    % solving motion
    sol_CAPL_dx = ode113(@(t,x) dx_CAPL(x),[0 t{idx}{end}], x0);
    [traj_CAPL_dx{idx}, ~] = deval(sol_CAPL_dx,cell2mat(t{idx})); % evaluation of solution
    sol_DPL_dx = ode113(@(t,x) dx_DPL(x),[0 t{idx}{end}], x0);
    [traj_DPL_dx{idx}, ~] = deval(sol_DPL_dx,cell2mat(t{idx})); % evaluation of solution
%     sol_CAPL_pi = ode113(@(t,x) policy_CAPL_{idx}(x),[0 t_{idx}{end}], x0);
%     [traj_CAPL_pi{idx}, ~] = deval(sol_CAPL_pi,cell2mat(t_{idx})); % evaluation of solution
%     sol_DPL_pi = ode113(@(t,x) policy_DPL_{idx}(x),[0 t_{idx}{end}], x0);
%     [traj_DPL_pi{idx}, ~] = deval(sol_DPL_pi,cell2mat(t_{idx})); % evaluation of solution
    % End-effector position
    pos_CAPL_dx{idx}=getPos(traj_CAPL_dx{idx}.');
    pos_DPL_dx{idx}=getPos(traj_DPL_dx{idx}.');
%     pos_CAPL_pi{idx}=getPos(traj_CAPL_pi{idx}.');
%     pos_DPL_pi{idx}=getPos(traj_DPL_pi{idx}.');
end
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Plot end-effector positions
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Plotting Results...\n');
figure(); hold on;
%for idx=1:NDem
for idx=1:Nidx
    subplot(4,4,idx);
    % plot
    plot3(p{idx}(:,1),p{idx}(:,2),p{idx}(:,3),'g'); hold on;
    plot3(pos_CAPL_dx{idx}(:,1),pos_CAPL_dx{idx}(:,2),pos_CAPL_dx{idx}(:,3),'b');
    plot3(pos_DPL_dx{idx}(:,1),pos_DPL_dx{idx}(:,2),pos_DPL_dx{idx}(:,3),'r');
    plot3(p{idx}(1,1),p{idx}(1,2),p{idx}(1,3),'k*');
    grid on; axis equal;
    axis equal;
    view(-60,20);
    title(int2str(idx));
end
%legend('data','policy','DMP','Location','best');
xlabel('x [m]'); ylabel('y [m]'); zlabel('z [m]');

%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Test policy in unseen constraint
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Compare with test dataset...\n');
for idx=1:numel(x_test)
    % Compute end-effector position:
    pos = getPos(cell2mat(x_test{idx}).'); % compute end-effector postion
    % Compute centre, normal and radious:
    [centre, radious, normal] = fit_3d_circle(pos(:,1),pos(:,2),pos(:,3));
    % Problem specific constants taken from data:
    x0 = x_test{idx}{1}; % initial configuration
    Kp = 0.1; % proportional gain
    % Constant matrices:
    W_A = blkdiag(normal.', normal.', normal.'); % constant gain matrix for the Constraint matrix
    W_b = -Kp*[W_A [-normal.'*centre; 0; 0]];
    % Definition of Constraint matrix and main task
    A = @(x) W_A*feval(Phi_A,x); % Constraint matrix as a function of configuration
    b = @(x) W_b*feval(Phi_b,x); % main task as a function of the configuration
    % Unconstrained policy
    %Phi = def_phi_4_cwm_sim(robot, centre, radious); % Get regressors
    Phi = def_phi_test(robot, centre);
    policy_CAPL = def_weighted_linear_model(model_CAPL, Phi);
    policy_DPL = def_weighted_linear_model(model_DPL, Phi);
    % Constrained policy
    dx_CAPL = def_constrained_policy(A, b, policy_CAPL);
    dx_DPL = def_constrained_policy(A, b, policy_DPL);
    % solving motion
    sol_CAPL_dx = ode113(@(t,x) dx_CAPL(x),[0 t_test{idx}{end}], x0);
    [traj_CAPL_dx, ~] = deval(sol_CAPL_dx,cell2mat(t_test{idx})); % evaluation of solution
    sol_DPL_dx = ode113(@(t,x) dx_DPL(x),[0 t_test{idx}{end}], x0);
    [traj_DPL_dx, ~] = deval(sol_DPL_dx,cell2mat(t_test{idx})); % evaluation of solution
    % End-effector position
    pos_CAPL_dx=getPos(traj_CAPL_dx.');
    pos_DPL_dx=getPos(traj_DPL_dx.');
    % Plot
    figure();
    plot3(pos(:,1),pos(:,2),pos(:,3),'g'); hold on;
    plot3(pos_CAPL_dx(:,1),pos_CAPL_dx(:,2),pos_CAPL_dx(:,3),'b');
    plot3(pos_DPL_dx(:,1),pos_DPL_dx(:,2),pos_DPL_dx(:,3),'r');
    grid on; axis equal;
    axis equal;
    view(-60,20);
end
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------


%% Plot end-effector positions
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
% fprintf(1,'Plotting Results...\n');
% figure(); hold on;
% %for idx=1:NDem
% for idx=1:Nidx
%     % plot
%     plot3(p{idx}(:,1),p{idx}(:,2),p{idx}(:,3),'g'); hold on;
%     plot3(pos_CAPL_dx{idx}(:,1),pos_CAPL_dx{idx}(:,2),pos_CAPL_dx{idx}(:,3),'b');
%     plot3(pos_DPL_dx{idx}(:,1),pos_DPL_dx{idx}(:,2),pos_DPL_dx{idx}(:,3),'r');
%     grid on; axis equal;
% end
% legend('data','policy','DMP','Location','best');
% xlabel('x [m]'); ylabel('y [m]'); zlabel('z [m]');
% grid on;
% axis equal;
% view(-60,20);
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------


%% Plot end-effector positions
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
% fprintf(1,'Plotting Results...\n');
% figure(); hold on;
% p_ = p(plot_idx);
% %for idx=1:NDem
% for idx=1:Nidx
%     subplot(3,4,idx);
%     % plot
%     plot3(p_{idx}(:,1),p_{idx}(:,2),p_{idx}(:,3),'g'); hold on;
%     plot3(pos_CAPL_pi{idx}(:,1),pos_CAPL_pi{idx}(:,2),pos_CAPL_pi{idx}(:,3),'b');
%     plot3(pos_DPL_pi{idx}(:,1),pos_DPL_pi{idx}(:,2),pos_DPL_pi{idx}(:,3),'r');
%     grid on;
% axis equal;
% end
% legend('data','policy','DMP','Location','best');
% xlabel('x [m]'); ylabel('y [m]'); zlabel('z [m]');
% grid on;
% axis equal;
% view(-60,20);
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
