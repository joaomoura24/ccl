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
file_name = 'data_ref_sep2';
file_ext = '.mat';
load(strcat(file_name,file_ext));
NDem = length(x); % number of demonstrations
corrPos = @(xs) fliplr(xs);
corrVel = @(xs) fliplr(cellfun(@(x) -1.*x,xs,'un',0));
% for idx=[1,13,14,15]
%     x{idx} = corrPos(x{idx});
%     u{idx} = corrVel(u{idx});
% end
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Initialize roobot model and the Regressors for the constraint and main task
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Defining robot model ...\n');
DH = [0.0, 0.31, 0.0, pi/2; % Robot Kinematic model specified by the Denavit-Hartenberg
      0.0, 0.0, 0.0, -pi/2;
      0.0, 0.4, 0.0, -pi/2;
      0.0, 0.0, 0.0, pi/2;
      0.0, 0.39, 0.0, pi/2;
      0.0, 0.0, 0.0, -pi/2;
      0.0, 0.21-0.132, 0.0, 0.0];
robot = SerialLink(DH); % Peters Cork robotics library has to be installed
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Compute centres and operational trajectory
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Compute end-effector positions ...\n');
p = cell(1, NDem); % end-effector cartesian position in global frame
dp = cell(1, NDem); % end-effector cartesian velocities in global frame
getPos = @(q) transl(robot.fkine(q)); % compute end-effector postion
for idx=1:NDem
    p{idx} = downsample(getPos(cell2mat(x{idx}).'),15); % compute end-effector postion
    dp{idx} = 2.*[zeros(1,3); diff(p{idx})];
end
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Plot end-effector positions
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Plotting Results...\n');
figure(); hold on;
col=jet(NDem);
%col = col(randperm(NDem),:);
legendInfo = cell(1, NDem);
for idx=1:NDem
    % plot
    plot3(p{idx}(:,1),p{idx}(:,2),p{idx}(:,3),'color',col(idx,:), 'LineWidth', 3);
    legendInfo{idx} = strcat('dem. ',int2str(idx)); % or whatever is appropriate
    grid on;
    axis image;
end
legend(legendInfo);
xlabel('x [m]');
ylabel('y [m]');
zlabel('z [m]');
view(50,25);
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%% Plot end-effector positions
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Plotting Results...\n');
figure(); hold on;
col=jet(NDem);
%col = col(randperm(NDem),:);
legendInfo = cell(1, NDem);
for idx=1:NDem
    % plot
    quiver3(p{idx}(:,1),p{idx}(:,2),p{idx}(:,3),dp{idx}(:,1),dp{idx}(:,2),dp{idx}(:,3),'color',col(idx,:), 'LineWidth', 3, 'AutoScaleFactor', 2);
    legendInfo{idx} = strcat('dem. ',int2str(idx)); % or whatever is appropriate
    grid on;
    axis image;
end
legend(legendInfo);
xlabel('x [m]');
ylabel('y [m]');
zlabel('z [m]');
view(50,25);
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Remove path
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
rmpath(genpath('../'));
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------