function functionHandle = def_phi_test(robotHandle, c_G)
% Libraries required: Peter Corke's Robotics library (MatLab add-on)
% 
% See also: def_phia_4_spm, def_phib_4_spm_sim, def_phia_4_spm

% Author: Joao Moura
% Edinburgh Centre for Robotics, Edinburgh, UK
% email address: Joao.Moura@ed.ac.uk
% Website: http://www.edinburgh-robotics.org/students/joao-moura
% March 2018; Last revision: 12-Mar-2018

%------------- BEGIN CODE --------------
functionHandle = @Phi;
function output = Phi(q)
     %J = robotHandle.jacobe(q); % Robot Jacobian in the end-effector frame.
     %Jtask = J(1:2,:); % Jacobian for the x and y coordinates - perpendicular plane to the end-effector.
     %T = robotHandle.fkine(q); % end-effector homogeneous transformation
     %tT = reshape(transl(T),[],1); % end-effector position
     %Jtask_inv = pinv(Jtask); % Pseudo-inverse of the task Jacobian.
     %N = eye(length(q)) - (Jtask_inv*Jtask); % Compute null-space projection matrix for given configuration.
%     Phi_kappa = kron([tT.' c_G.'],eye(3)); % Regressors for the secondary task.
%     Phi_gamma = kron([q.' 1],eye(length(q))); % Regressors for the third task.
    %c_ro = feval(getC_ro(robotHandle, c_G),q); % centre of the circular motion to the end-effector relative to the end-effector frame
    %output = Jtask_inv*kron([tT.' c_G.' 1],eye(2)); %  N*Phi_gamma
    %output = kron([q.' 1],eye(length(q)));
    output = kron([q.' c_G.' 1],eye(7));
    %output = Jtask_inv*kron(c_ro',eye(2));
    output(end,:) = 0;
    %output = kron([q' c_G' 1],eye(6));
%     function functionHandle = getC_ro(robotHandle, c_G)
%         functionHandle = @c_ro;
%         function output = c_ro(q)
%             T = robotHandle.fkine(q); % end-effector homogeneous transformation
%             tT = reshape(transl(T),[],1); % end-effector position
%             R = t2r(T); % end-effector orientation (rotation matrix)
%             centre = R.'*(c_G - tT); % distance of the end-effector position and centre position rotated for end-effector frame
%             output = centre(1:2);
%         end
%     end
end
%------------- END OF CODE --------------
end
