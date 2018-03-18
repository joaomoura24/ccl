function functionHandle = def_phi_circ()

% Author: Joao Moura
% Edinburgh Centre for Robotics, Edinburgh, UK
% email address: Joao.Moura@ed.ac.uk
% Website: http://www.edinburgh-robotics.org/students/joao-moura
% March 2018; Last revision: 13-Mar-2018

%------------- BEGIN CODE --------------
functionHandle = @Phi;
function output = Phi(q)
    %output = kron([q.' c_G.' 1],eye(3));
    output = kron(q.',eye(3));
    %output = diag(q);
end
%------------- END OF CODE --------------
end
