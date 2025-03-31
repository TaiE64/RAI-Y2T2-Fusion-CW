clc; clear;

%% === 加载数据和标定参数 ===
load('calibration_params.mat');             % 包含 gyro_bias, accel_bias, tof_bias
load('task1_2.mat');           % 包含 out 结构体

%% === 传感器数据提取 ===
gyro  = squeeze(out.Sensor_GYRO.signals.values)';   % [N x 3]
accel = squeeze(out.Sensor_ACCEL.signals.values)';  % [N x 3]

tof1_data = squeeze(out.Sensor_ToF1.signals.values);
tof2_data = squeeze(out.Sensor_ToF2.signals.values);
tof3_data = squeeze(out.Sensor_ToF3.signals.values);
tof_data = [tof3_data(:,1), tof2_data(:,1), tof1_data(:,1)];

time    = out.GT_time.time;
GT_pos  = out.GT_position.signals.values(:,1:2);
GT_rot  = out.GT_rotation.signals.values;
GT_rot(1,:) = GT_rot(2,:);  % 修复 NaN

GT_q = quaternion(GT_rot);
eul  = eulerd(GT_q, 'ZYX', 'frame');
GT_yaw = deg2rad(eul(:,1)) + pi;
GT_yaw = unwrap(GT_yaw);
init_cor = GT_pos(1,:);
init_rot = GT_yaw(1);

%% === Bias 标定 ===
gyro_calibrated  = gyro - calibration_params.gyro_bias;
accel_calibrated = accel - calibration_params.accel_bias;
tof_calibrated   = tof_data - calibration_params.tof_bias;

%% === EKF 初始化 ===
N = length(time);
fs = 1 / mean(diff(time));

x_est = zeros(N, 5);  % [x, y, vx, vy, yaw]
x_est(1,:) = [init_cor, 0, 0, init_rot];
P = diag([0.01, 0.01, 0.01, 0.01, 0]);
Q = diag([0.0001, 0.0001, 0.001, 0.001, 0]);  % 过程噪声
R = eye(3) * 5;                             % ToF观测噪声

%% === 高通滤波器参数 ===
tau = 6;  % 高通滤波时间常数
a_world_corrected = zeros(N,2);
a_w_raw_prev = [0, 0];

for i = 2:N
    dt = time(i) - time(i-1);
    alpha_hpf = tau / (tau + dt);

    %% --- 预测步 ---
    a_b = [accel_calibrated(i,2), accel_calibrated(i,3)];
    yaw = x_est(i-1,5);
    R_yaw = [sin(yaw), cos(yaw); -cos(yaw), sin(yaw)];
    R_yaw = [sin(GT_yaw(i)), cos(GT_yaw(i)); -cos(GT_yaw(i)), sin(GT_yaw(i))];
    a_w_raw = (R_yaw * a_b')'; %得到当前时间步下Acc_world

    if i == 2
        a_world_corrected(i,:) = a_w_raw;
    else
        a_world_corrected(i,:) = alpha_hpf * (a_world_corrected(i-1,:) + a_w_raw - a_w_raw_prev);
    end
    a_w_raw_prev = a_w_raw;
    a_w = a_world_corrected(i,:);

    % 状态预测
    vx = x_est(i-1,3) + a_w(1) * dt;
    vy = x_est(i-1,4) + a_w(2) * dt;

    x_pred = x_est(i-1,:);
    x_pred(1) = x_pred(1) + vx * dt+0.5*a_w(1) * dt^2;
    x_pred(2) = x_pred(2) + vy * dt+0.5*a_w(2) * dt^2;
    x_pred(3) = vx;
    x_pred(4) = vy;
    x_pred(5) = x_pred(5) + gyro_calibrated(i,1) * dt;

    % 线性化状态转移矩阵
    F = eye(5);
    F(1,3) = dt;
    F(2,4) = dt;
    P = F * P * F' + Q;

    %% --- 观测模型 ---
    [z_pred, H] = measurementModel(x_pred);
    z_meas = tof_calibrated(i,:)';
    residual = z_meas - z_pred
    S = H * P * H' + R;
    K = P * H' / S;
    x_est(i,:) = x_pred + (K * residual)';
    P = (eye(5) - K * H) * P;
end

%% === 可视化 ===
figure;

% 1. Trajectory XY plot
subplot(3,2,1);
plot(GT_pos(:,1), GT_pos(:,2), 'g--', 'LineWidth', 2); hold on;
plot(x_est(:,1), x_est(:,2), 'r-', 'LineWidth', 1.5);
legend('Ground Truth', 'EKF Estimate');
xlabel('X [m]'); ylabel('Y [m]');
title('Trajectory with EKF Fusion');
axis equal; grid on;

% 2. Y-axis velocity
subplot(3,2,2);
plot(time, x_est(:,4), 'r-', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('Y-axis Speed [m/s]');
title('Y-axis Velocity from EKF');
grid on;

% 3. World-frame Y acceleration
subplot(3,2,3);
plot(time, a_world_corrected(:,2), 'b-', 'LineWidth', 1.2);
xlabel('Time [s]'); ylabel('Filtered Accel Y [m/s^2]');
title('World-frame Y Acceleration');
grid on;

% 4. Ground Truth position over time
subplot(3,2,4);
plot(time, GT_pos(:,1), 'b-', 'LineWidth', 1.2); hold on;
plot(time, GT_pos(:,2), 'r-', 'LineWidth', 1.2);
xlabel('Time [s]'); ylabel('Position [m]');
legend('GT X', 'GT Y');
title('Ground Truth Position over Time');
grid on;

% 5. Estimated position over time
subplot(3,2,5);
plot(time, x_est(:,1), 'b-', 'LineWidth', 1.2); hold on;
plot(time, x_est(:,2), 'r-', 'LineWidth', 1.2);
xlabel('Time [s]'); ylabel('Position [m]');
legend('Est X', 'Est Y');
title('Estimated Position over Time');
grid on;

% 6. Yaw comparison
subplot(3,2,6);
plot(time, x_est(:,5), 'r-', 'LineWidth', 1.2); hold on;
plot(time, GT_yaw, 'b--', 'LineWidth', 1.2);
xlabel('Time [s]'); ylabel('Yaw [rad]');
legend('Estimated', 'Ground Truth');
title('Yaw Comparison over Time');
grid on;

sgtitle('EKF Sensor Fusion Visualization'); % 总标题

%% === Measurement Model Function ===
function [z_pred, H] = measurementModel(x)
    % 使用数值法计算 ToF 观测模型的预测值和雅可比矩阵
    % 输入:
    %   x: [x, y, vx, vy, yaw]
    % 输出:
    %   z_pred: [tof_right; tof_front; tof_left]
    %   H: 3x5 雅可比矩阵

    % 设置传感器朝向（单位为弧度）：右, 前, 左（与 TEST 中 tof_data 顺序一致）
    sensor_angles = [pi/2, 0, -pi/2];

    z_pred = zeros(3,1);
    H = zeros(3,5);

    for i = 1:3
        [z_pred(i), H_row] = h_tof_numeric(x, sensor_angles(i));
        H(i,:) = H_row;
    end
end

function [z, H] = h_tof_numeric(x, sensor_angle)
    % 数值计算单个ToF观测值及其雅可比（来自untitled2）
    pos = x(1:2);
    theta = x(5);
    beta = theta + sensor_angle;

    wall_vals = [1.18, -1.18, 1.2, -1.2];  % 与 TEST.m 的墙体匹配
    t_candidates = [];

    if abs(cos(beta)) > 1e-5
        for c = wall_vals(1:2)  % vertical walls
            t = (c - pos(1)) / cos(beta);
            if t > 0
                t_candidates(end+1) = t;
            end
        end
    end
    if abs(sin(beta)) > 1e-5
        for c = wall_vals(3:4)  % horizontal walls
            t = (c - pos(2)) / sin(beta);
            if t > 0
                t_candidates(end+1) = t;
            end
        end
    end

    if isempty(t_candidates)
        z = inf;
    else
        z = min(t_candidates);
    end

    % 数值求导 Jacobian
    delta = 1e-5;
    grad = zeros(1,3);
    for i = 1:3  % 对 x, y, yaw 求导
        x_pert = x;
        idx = [1, 2, 5];  % 只对 x, y, yaw 方向加扰动
        x_pert(idx(i)) = x_pert(idx(i)) + delta;

        % 重新计算观测值
        pos_p = x_pert(1:2);
        theta_p = x_pert(5);
        beta_p = theta_p + sensor_angle;

        t_p = [];

        if abs(cos(beta_p)) > 1e-5
            for c = wall_vals(1:2)
                tp = (c - pos_p(1)) / cos(beta_p);
                if tp > 0
                    t_p(end+1) = tp;
                end
            end
        end
        if abs(sin(beta_p)) > 1e-5
            for c = wall_vals(3:4)
                tp = (c - pos_p(2)) / sin(beta_p);
                if tp > 0
                    t_p(end+1) = tp;
                end
            end
        end

        if isempty(t_p)
            zp = inf;
        else
            zp = min(t_p);
        end

        grad(i) = (zp - z) / delta;
    end

    H = [grad(1), grad(2), 0, 0, grad(3)];
end
