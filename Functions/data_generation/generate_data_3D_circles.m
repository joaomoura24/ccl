% Generates circular trajectories of a particle, outputting the Cartesian
% position and velocity of the particle.

% Author: Joao Moura
% Edinburgh Centre for Robotics, Edinburgh, UK
% email address: Joao.Moura@ed.ac.uk
% Website: http://www.edinburgh-robotics.org/students/joao-moura
% March 2018; Last revision: 18-Oct-2017

%% User Input
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
freq = 50; % frequency of the data
tf = 5; % duration of the simulation
centers = {[0;0;0], [0;0;0], [0;0;0], [0;0;0]};
radiuses = {1, 1, 1, 1};
normals = {[0; cosd(45); sind(45)]...
           [cosd(45); 0; sind(45)]...
           [0; -cosd(45); sind(45)]...
           [0; 0; 1]};
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------


%% Generate data
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Generating data ...\n');
N = tf*freq;
NDem = numel(radiuses);
theta = linspace(0,2*pi,N + 1);
% Initialize cells for storing data:
x = cell(1, NDem); % configurations
u = cell(1, NDem); % actions
t = cell(1, NDem); % configuration data
% Plot
figure(); hold on;
for idx=1:NDem
    % Get geometrix information:
    center = centers{idx};
    radius = radiuses{idx};
    normal = normals{idx};
    % Null space base of the constraint:
    v=null(normal.');
    % Data points:
    points=repmat(center,1,size(theta,2))+radius*(v(:,1)*cos(theta)+v(:,2)*sin(theta));
    % plot result:
    plot3(points(1,:),points(2,:),points(3,:),'*r');
    % Store data points as cell:
    u{idx} = num2cell(diff(points,1,2),1); % save actions: cartesian velocities
    x{idx} = num2cell(points(:,1:end-1),1); % save state: cartesian positions
    t{idx} = num2cell(linspace(0,tf,N),1); % demonstration time
end
% Plot
xlabel('x'); ylabel('y'); zlabel('z');
axis square; grid on; view(130,25);
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Save data to file
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
save('data_3D_circles.mat','x','u','t');
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------