clc; clear;
%% === Load Data ===
straight_cali_data = load('trainingData/calib2_straight.mat');
rotate_cali_data = load('trainingData/calib1_rotate.mat');

%% === Parameter Settings ===
T_static = 60;                % Duration of static segment [s]
Fs_imu = 200;                 % IMU sampling rate
Fs_mag = 200;                 % Magnetometer sampling rate
Fs_tof = 200;                 % ToF sampling rate

N_imu = round(Fs_imu * T_static);
N_mag = round(Fs_mag * T_static);
N_tof = round(Fs_tof * T_static);

%% === Gyroscope Bias Calibration ===
gyro_data = squeeze(straight_cali_data.out.Sensor_GYRO.signals.values)';  % [N x 3]
gyro_static = gyro_data(1:N_imu, :);
gyro_bias = mean(gyro_static, 1);

%% === Accelerometer Bias Calibration ===
accel_data = squeeze(straight_cali_data.out.Sensor_ACCEL.signals.values)';  % [N x 3]
accel_static = accel_data(1:N_imu, :);
true_gravity = [9.81, 0, 0];    % Assume gravity aligned with X axis
accel_bias = mean(accel_static, 1) - true_gravity;

%% === Magnetometer Bias Calibration ===
mag_data = squeeze(rotate_cali_data.out.Sensor_MAG.signals.values)';     % [N x 3]
[mag_calibrated, mag_offset, mag_gain, mag_rotM] = calibrateMagnetometer(mag_data);

%% === Visualization: Raw vs. Calibrated Ellipsoid ===
figure;
subplot(1,2,1);
scatter3(mag_data(:,1), mag_data(:,2), mag_data(:,3), 5, 'filled');
title('Raw Magnetometer');
xlabel('X'); ylabel('Y'); zlabel('Z');
axis equal; grid on;

subplot(1,2,2);
scatter3(mag_calibrated(:,1), mag_calibrated(:,2), mag_calibrated(:,3), 5, 'filled');
title('Calibrated Magnetometer');
xlabel('X'); ylabel('Y'); zlabel('Z');
axis equal; grid on;

%% === Output Calibration Parameters ===
disp('--- Magnetometer Calibration Parameters ---');
disp(['Offset (bias): ', mat2str(mag_offset, 4)]);
disp(['Gain (scale):  ', mat2str(mag_gain, 4)]);
disp('Rotation Matrix (soft-iron compensation): ');
disp(mag_rotM);

%% === ToF Sensor Bias Calibration (first N_tof distance values only) ===
tof1_data = squeeze(straight_cali_data.out.Sensor_ToF1.signals.values);  % [N x 4]
tof2_data = squeeze(straight_cali_data.out.Sensor_ToF2.signals.values);
tof3_data = squeeze(straight_cali_data.out.Sensor_ToF3.signals.values);

tof1_mean = mean(tof1_data(1:N_tof, 1));  % Distance column
tof2_mean = mean(tof2_data(1:N_tof, 1));
tof3_mean = mean(tof3_data(1:N_tof, 1));

%% === Ground Truth Trajectory and Orientation Extraction ===
GT_position = squeeze(straight_cali_data.out.GT_position.signals.values);   % [N x 3]
GT_orientation = squeeze(straight_cali_data.out.GT_rotation.signals.values); % [N x 4] (w x y z)
xy = GT_position(:, 1:2);  % Extract XY plane position

% Process quaternion and extract yaw
q = quaternion(GT_orientation);                  
eul = eulerd(q, 'ZYX', 'frame');                 % Euler angles [yaw pitch roll] (°)
yaw_rad = deg2rad(eul(:,1)) + pi;                % Convert to radians
dx = cos(yaw_rad); dy = sin(yaw_rad);            % Direction vector

%% === Visualization: Trajectory + Orientation Arrows ===
figure;
h1 = plot(xy(:,1), xy(:,2), 'k--'); hold on;
step = 5;
h2 = quiver(xy(1:step:end,1), xy(1:step:end,2), dx(1:step:end), dy(1:step:end), 0.3, 'r');
axis equal;
xlabel('X [m]');
ylabel('Y [m]');
title('2D Ground Truth Trajectory with Orientation');
legend([h1, h2], {'Trajectory', 'Orientation'});

%% === ToF Bias = Measured Distance - Actual Distance (estimated from GT start pose) ===
start_position = GT_position(1,:);
% Small heading offset (2–3 degrees) is considered in estimation.
d3_gt = abs((-1.2 - start_position(1))/sin(yaw_rad(1)));   % Left wall distance
d2_gt = abs((1.2  - start_position(2))/sin(yaw_rad(1)));   % Center (front) wall
d1_gt = abs((1.2  - start_position(1))/sin(yaw_rad(1)));   % Right wall distance
tof1_bias = tof1_mean - d1_gt;
tof2_bias = tof2_mean - d2_gt;
tof3_bias = tof3_mean - d3_gt;
tofbias = [tof3_bias, tof2_bias, tof1_bias];  % [Left Center Right]

%% === Output All Bias Results ===
disp('--- Sensor Bias Summary ---');
disp(['Gyroscope Bias      : ', mat2str(gyro_bias, 4)]);
disp(['Accelerometer Bias  : ', mat2str(accel_bias, 4)]);
disp(['ToF Bias [L C R]    : ', mat2str(tofbias, 4)]);

% Convert all quaternions to Euler angles
q = quaternion(GT_orientation);               
eul_all = eulerd(q, 'ZYX', 'frame');   % [yaw, pitch, roll] in degrees

% Store Euler angles
yaw   = eul_all(:, 1);   % Column 1: yaw (Z-axis)
pitch = eul_all(:, 2);   % Column 2: pitch (Y-axis)
roll  = eul_all(:, 3);   % Column 3: roll (X-axis)

%% === Save All Calibration Parameters to .mat File ===
calibration_params = struct();

calibration_params.gyro_bias  = gyro_bias;
calibration_params.accel_bias = accel_bias;

calibration_params.mag_offset = mag_offset;
calibration_params.mag_gain   = mag_gain;
calibration_params.mag_rotM   = mag_rotM;

calibration_params.tof_bias   = tofbias;

save('calibration_params.mat', 'calibration_params');
disp('--- Calibration parameters saved to calibration_params.mat ---');

%% Apply Magnetometer Calibration
% Input: raw data, known bias parameters
function mag_calibrated = applyMagCalibration(mag_raw, offset, gain, rotM)
    mag_centered = mag_raw - offset';
    mag_rotated = (rotM' * mag_centered')';
    mag_calibrated = mag_rotated ./ gain';
end

%% Estimate Magnetometer Bias Parameters
function [magCalibrated, offset, gain, rotM] = calibrateMagnetometer(mag_raw)
    [offset, gain, rotM] = ellipsoid_fit(mag_raw, 0);  
    [gain, rotM] = refine_3D_fit(gain, rotM);
  
    mag_centered = mag_raw - offset';                   
    mag_rotated = (rotM' * mag_centered')';            
    magCalibrated = mag_rotated ./ gain';              
end

% Ellipsoid Fitting for Bias Estimation
function [ofs, gain, rotM] = ellipsoid_fit(XYZ, flag)
    x = XYZ(:, 1); y = XYZ(:, 2); z = XYZ(:, 3);
    if nargin < 2
        flag = 0;
    end
    switch flag
        case 0
            D = [x.^2, y.^2, z.^2, 2*x.*y, 2*x.*z, 2*y.*z, 2*x, 2*y, 2*z];
        case 1
            D = [x.^2, y.^2, z.^2, 2*x, 2*y, 2*z];
        case 2
            D = [x.^2 + y.^2, z.^2, 2*x, 2*y, 2*z];
        case 3
            D = [x.^2 + z.^2, y.^2, 2*x, 2*y, 2*z];
        case 4
            D = [y.^2 + z.^2, x.^2, 2*x, 2*y, 2*z];
        case 5
            D = [x.^2 + y.^2 + z.^2, 2*x, 2*y, 2*z];
    end
    v = (D' * D) \ (D' * ones(length(x), 1));
    if flag == 0
        A = [v(1), v(4), v(5), v(7);
             v(4), v(2), v(6), v(8);
             v(5), v(6), v(3), v(9);
             v(7), v(8), v(9), -1];
        ofs = -A(1:3, 1:3) \ [v(7); v(8); v(9)];
        T = eye(4); T(4, 1:3) = ofs';
        Atrans = T * A * T';
        [rotM, ev] = eig(Atrans(1:3, 1:3) / -Atrans(4, 4));
        gain = sqrt(1 ./ diag(ev));
    else
        switch flag
            case 1, v = [v(1), v(2), v(3), 0, 0, 0, v(4), v(5), v(6)];
            case 2, v = [v(1), v(1), v(2), 0, 0, 0, v(3), v(4), v(5)];
            case 3, v = [v(1), v(2), v(1), 0, 0, 0, v(3), v(4), v(5)];
            case 4, v = [v(2), v(1), v(1), 0, 0, 0, v(3), v(4), v(5)];
            case 5, v = [v(1), v(1), v(1), 0, 0, 0, v(2), v(3), v(4)];
        end
        ofs = -(v(1:3) .\ v(7:9))';
        rotM = eye(3);
        g = 1 + (v(7)^2 / v(1) + v(8)^2 / v(2) + v(9)^2 / v(3));
        gain = sqrt(g ./ v(1:3))';
    end
end

% Ensure Gain and Rotation Matrix Consistency
function [gain, rotM] = refine_3D_fit(gain, rotM)
    [~, idx] = max(abs(rotM), [], 'all', 'linear');
    [rm, cm] = ind2sub([3, 3], idx);
    if rm ~= cm
        rotM(:, [rm, cm]) = rotM(:, [cm, rm]);
        [gain(rm), gain(cm)] = deal(gain(cm), gain(rm));
    end
    i = setdiff(1:3, rm);
    sub = abs(rotM(i, i));
    [~, idx] = max(sub, [], 'all', 'linear');
    [rsub, csub] = ind2sub([2, 2], idx);
    if rsub ~= csub
        rotM(:, i([rsub, csub])) = rotM(:, i([csub, rsub]));
        [gain(i([rsub, csub]))] = deal(gain(i([csub, rsub])));
    end
    for i = 1:3
        if rotM(i, i) < 0
            rotM(:, i) = -rotM(:, i);
        end
    end
end
