clear;clc;
load('task2_5.mat');                 % Load Ground Truth data
Pred = load('Task2_5_Estimation.mat');       % Load EKF estimated results

% Extract GT position and yaw
time    = out.GT_time.time;
GT_pos  = out.GT_position.signals.values(:,1:2);
GT_rot  = out.GT_rotation.signals.values;
GT_rot(1,:) = GT_rot(2,:);           % Fix NaN at the first frame

GT_q   = quaternion(GT_rot);        % Convert rotation to quaternion
eul    = eulerd(GT_q, 'ZYX', 'frame'); 
GT_yaw = unwrap(deg2rad(eul(:,1)) + pi);  % Get yaw angle in radians

% Extract EKF estimated position and yaw
Pred_pos = Pred.x_est(:,1:2);
Pred_yaw = Pred.x_est(:,5);

% Combine into [x, y, yaw] matrices
gt  = [GT_pos, GT_yaw];
est = [Pred_pos, Pred_yaw];

% Compare and visualize trajectories
compare_xyyaw_trajectories(est, gt);

%% === Compare function ===
function compare_xyyaw_trajectories(est, gt)
    % est and gt should be N×3 matrices: [x, y, yaw] (unit: meter + rad)
    assert(size(est,2) == 3 && size(gt,2) == 3, ...
        'Inputs must be N×3 matrices of [x, y, yaw]');

    % Convert to rigidtform3d trajectories
    estTraj = xyyaw_to_rigid(est);
    gtTraj = xyyaw_to_rigid(gt);

    % Compute trajectory difference
    mtrics = compareTrajectories(gtTraj, estTraj);
    disp(['Absolute RMSE for key frame location (m): ', num2str(mtrics.AbsoluteRMSE(2))]);

    % Plot error metrics
    figure;
    ax = plot(mtrics, "absolute-translation");

    disp('--- Debug Info: compareTrajectories executed successfully. ---');
end

%% === Convert [x, y, yaw] to rigidtform3d array ===
function traj = xyyaw_to_rigid(data)
    numFrames = size(data, 1);
    traj = rigidtform3d.empty(numFrames, 0);

    for i = 1:numFrames
        x = data(i,1);
        y = data(i,2);
        yaw = data(i,3);  % yaw in radians

        % Rotation matrix around Z-axis
        R = [cos(yaw), -sin(yaw), 0;
             sin(yaw),  cos(yaw), 0;
             0, 0, 1];
        
        % Translation vector
        t = [x, y, 0];

        % Construct rigid transformation
        traj(i) = rigidtform3d(R, t);
    end
end
