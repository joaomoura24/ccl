% EXTRACT_DATA_KUKA_WIPING.m - Extracts the joint positions  and the time 
% from the mat files that were previously generated from the ROS bag files
%
% Other m-files required: bag2matlab.m - to convert ROS bag files into mat
% files

% Author: Joao Moura
% Edinburgh Centre for Robotics, Edinburgh, UK
% email address: Joao.Moura@ed.ac.uk
% Website: http://www.edinburgh-robotics.org/students/joao-moura
% October 2017; Last revision: 25-Oct-2017

%% User Input
files_directory = '../../../Leo_code_test/joao_library/demonstrations_mat/';
N_cut = 100; % Initial N samples to be ignored due to initial adaptation to path.
data_file_name = 'data_smoth.mat';
NDem = 12; % Number of demonstrations to store (< number of files in dir).
K_mov_mean = 100; % Inteval for moving average to smoth the trainning data.

%% Get data
file_names = dir(strcat(files_directory,'2017*.mat')); % get data from real
% robot (previously converted from bag file to *.mat file)
Nfiles = length(file_names); % number of *.mat files in the specified directory
% Condition to assert if the NDem is not > number of files
if NDem > Nfiles; NDem = Nfiles; end
x = cell(1, NDem); % store state (robot configuration)
u = cell(1, NDem); % store input (configuration velocity)
t = cell(1, NDem); % store time
% Loop to extract and save data
for i=1:NDem
    file_name = file_names(i).name;
    fprintf(1,'Processing file %i: %s ...\n',i,file_name);
    load(strcat(files_directory,file_name)); % Get data from 1 demonstration.
    q_i = movmean(cell2mat(data_struct.jointPosition),K_mov_mean); % Joint angles.
    dt_i = data_struct.dT(2:end);
    x{i} = num2cell(q_i(2:end,:).',1); % save state: joint angles
    u{i} = num2cell((diff(q_i)./dt_i).',1); % save input: joint velocities
    t{i} = num2cell(cumsum(dt_i).',1); % demonstration time
    % cut initial samples
    x{i} = x{i}(N_cut:end);
    u{i} = u{i}(N_cut:end);
    t{i} = gsubtract(t{i}(N_cut:end),t{i}{N_cut});
end
save(data_file_name,'x','u','t');