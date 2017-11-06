function functionHandle = def_phia_4_spm_new(robotHandle)
% Defines suitable regressors for the constraint matrix for a surface perpendicular motion.
%
% Consider a Constraint matrix defined as a linear combination of regressors:
%
%         A(x) = W_A * Phi_A(x),
%
% where W_A is a matrix of weights, and Phi_A(x) is a matrix of regressors.
% def_phia_4_spm_new returns a MatLab function handle to a set of regressors
% suitable for the constraint of moving the robot end-effector in contact 
% and perpendicular to a surface.
% This regressors are a function of the robot configuration - column vector.
%
% Syntax:  functionHandle = def_phia_4_spm_new(robotHandle)
%
% Inputs:
%    robotHandle - Peter Corke's Serial-link robot class
%
% Outputs:
%    functionHandle - MatLab function handle with robot configuration 
%                     (column vector) as input
%
% Example: 
%     % Robot Kinematic model specified by the Denavit-Hartenberg.
%     DH = [0.0, 0.31, 0.0, pi/2;
%           0.0, 0.0, 0.0, -pi/2;
%           0.0, 0.4, 0.0, -pi/2;
%           0.0, 0.0, 0.0, pi/2;
%           0.0, 0.39, 0.0, pi/2;
%           0.0, 0.0, 0.0, -pi/2;
%           0.0, 0.21-0.132, 0.0, 0.0];
%     % Peters Cork robotics library has to be installed:
%     robot = SerialLink(DH);
%     % Phi_A(x): vector of regressors for the Constraint matrix as a function
%     % of the state
%     n = [0; 0; 1]; % Cartesian normal of the constraint surface.
%     W_A = blkdiag(n.', n.', n.'); % Constant gain matrix for the Constraint matrix.
%     Phi_A = def_phia_4_spm_new(robot);
%     % Constraint matrix as a function of configuration.
%     A = @(x) W_A*PhiA(x);
%     % Constraint matrix for given robot arm configuration:
%     disp(A([0;0;0;pi/2;0;-pi/2;0]));
%
% Libraries required: Peter Corke's Robotics library (MatLab add-on)
% 
% See also: def_phib_4_spm_sim, def_phib_4_spm_exp, def_phi_4_cwm

% Author: Joao Moura
% Edinburgh Centre for Robotics, Edinburgh, UK
% email address: Joao.Moura@ed.ac.uk
% Website: http://www.edinburgh-robotics.org/students/joao-moura
% October 2017; Last revision: 28-Oct-2017

%------------- BEGIN CODE --------------
functionHandle = @Phi_A;
function output = Phi_A(q)
    J = robotHandle.jacobn(q); % Robot Jacobian in the end-effector reference frame.
    output = J(3:5,:);
end
%------------- END OF CODE --------------
end
