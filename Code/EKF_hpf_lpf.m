clc; clear;

%% === 加载数据和标定参数 ===
load('calibration_params.mat');      % 包含 gyro_bias, accel_bias, tof_bias
load('task1_1.mat');                 % 包含 out 结构体

%% === 提取传感器数据 ===
gyro  = squeeze(out.Sensor_GYRO.signals.values)';    % [N x 3]
accel = squeeze(out.Sensor_ACCEL.signals.values)';   % [N x 3]

tof1_data = squeeze(out.Sensor_ToF1.signals.values);
tof2_data = squeeze(out.Sensor_ToF2.signals.values);
tof3_data = squeeze(out.Sensor_ToF3.signals.values);
tof_data = [tof3_data(:,1), tof2_data(:,1), tof1_data(:,1)];  % [右, 前, 左]

time    = out.GT_time.time;
GT_pos  = out.GT_position.signals.values(:,1:2);
GT_rot  = out.GT_rotation.signals.values;
GT_rot(1,:) = GT_rot(2,:);  % 修复 NaN

GT_q   = quaternion(GT_rot);
eul    = eulerd(GT_q, 'ZYX', 'frame');
GT_yaw = unwrap(deg2rad(eul(:,1)) + pi);

init_cor = GT_pos(1,:);
init_rot = GT_yaw(1);

%% === Bias 校正 ===
gyro_calibrated  = gyro - calibration_params.gyro_bias;
accel_calibrated = accel - calibration_params.accel_bias;
calibration_params.tof_bias(:, [1 3])=calibration_params.tof_bias(:, [3 1]);
tof_calibrated   = tof_data - calibration_params.tof_bias;

%% === EKF 初始化 ===
N = length(time);
x_init = zeros(N, 5);  % [x, y, vx, vy, yaw]
x_init(1,:) = [init_cor, 0, 0, init_rot];

P = diag([0.01, 0.01, 0.01, 0.01, 0]);            % 初始协方差
Q = diag([0.0001, 0.0001, 0.001, 0.001, 0]);      % 过程噪声
R = eye(3) * 5;                                   % ToF 观测噪声

[x_est, P_est, a_world_corrected] = EKF(x_init, P, time, ...
    tof_calibrated, gyro_calibrated, accel_calibrated, Q, R, 6,0.02);

%% === 可视化结果 ===
figure;

% 1. XY轨迹
subplot(3,2,1);
plot(GT_pos(:,1), GT_pos(:,2), 'g--', 'LineWidth', 2); hold on;
plot(x_est(:,1), x_est(:,2), 'r-', 'LineWidth', 1.5);
legend('Ground Truth', 'EKF Estimate');
xlabel('X [m]'); ylabel('Y [m]');
title('Trajectory with EKF Fusion');
axis equal; grid on;

% 2. Y轴速度
subplot(3,2,2);
plot(time, x_est(:,4), 'r-', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('Y-axis Speed [m/s]');
title('Y-axis Velocity from EKF');
grid on;

% 3. 世界坐标下加速度
subplot(3,2,3);
plot(time, a_world_corrected(:,2), 'b-', 'LineWidth', 1.2);
xlabel('Time [s]'); ylabel('Filtered Accel Y [m/s^2]');
title('World-frame Y Acceleration');
grid on;

% 4. Ground Truth 位置随时间变化
subplot(3,2,4);
plot(time, GT_pos(:,1), 'b-', 'LineWidth', 1.2); hold on;
plot(time, GT_pos(:,2), 'r-', 'LineWidth', 1.2);
xlabel('Time [s]'); ylabel('Position [m]');
legend('GT X', 'GT Y');
title('Ground Truth Position over Time');
grid on;

% 5. 估计位置随时间变化
subplot(3,2,5);
plot(time, x_est(:,1), 'b-', 'LineWidth', 1.2); hold on;
plot(time, x_est(:,2), 'r-', 'LineWidth', 1.2);
xlabel('Time [s]'); ylabel('Position [m]');
legend('Est X', 'Est Y');
title('Estimated Position over Time');
grid on;

% 6. Yaw 比较
subplot(3,2,6);
plot(time, x_est(:,5), 'r-', 'LineWidth', 1.2); hold on;
plot(time, GT_yaw, 'b--', 'LineWidth', 1.2);
xlabel('Time [s]'); ylabel('Yaw [rad]');
legend('Estimated', 'Ground Truth');
title('Yaw Comparison over Time');
grid on;

sgtitle('EKF Sensor Fusion Visualization');

save('Prediction.mat', 'x_est');

%% === EKF 主函数 ===
function [x_est, P_est, a_world_corrected] = EKF(x_init, P_init, time, ToF, Gyro, Acc, Q, R, tau_hpf,tau_lpf)
    N = length(time);
    x_est = x_init;
    P = P_init;
    P_est = cell(N, 1);
    a_world_corrected = zeros(N, 2);
    a_w_raw_prev = [0, 0];

    % 初始化低通滤波
    pos_lpf = x_init(1,1:2);

    for i = 2:N
        dt = time(i) - time(i-1);
        alpha_hpf = tau_hpf / (tau_hpf + dt);
        alpha_lpf = tau_lpf / (tau_lpf + dt);  % 低通滤波系数

        %% --- 预测步骤 ---
        a_b = [Acc(i,2), Acc(i,3)];
        yaw = x_est(i-1,5);
        R_yaw = [sin(yaw), cos(yaw); -cos(yaw), sin(yaw)];
        a_w_raw = (R_yaw * a_b')';

        % 高通滤波（用于加速度）
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
        x_pred(1) = x_pred(1) + vx * dt + 0.5 * a_w(1) * dt^2;
        x_pred(2) = x_pred(2) + vy * dt + 0.5 * a_w(2) * dt^2;
        x_pred(3) = vx;
        x_pred(4) = vy;
        x_pred(5) = x_pred(5) + Gyro(i,1) * dt;

        % 线性化状态转移矩阵
        F = eye(5);
        F(1,3) = dt;
        F(2,4) = dt;
        P = F * P * F' + Q;

        %% --- 观测更新 ---
        [z_pred, H] = measurementModel(x_pred);
        z_meas = ToF(i,:)';
        residual = z_meas - z_pred;
        S = H * P * H' + R;
        K = P * H' / S;

        x_update = x_pred + (K * residual)';
        P = (eye(5) - K * H) * P;
        P_est{i-1} = P;

        %% === 位置低通滤波 ===
        if i == 2
            pos_lpf = x_update(1:2);  % 初始化
        else
            pos_lpf = alpha_lpf * pos_lpf + (1 - alpha_lpf) * x_update(1:2);
        end
        x_update(1:2) = pos_lpf;  % 替换为滤波后位置

        x_est(i,:) = x_update;
    end
end

function [z_pred, H] = measurementModel(x)
    % 使用数值法计算 ToF 观测模型的预测值和雅可比矩阵
    % 输入:
    %   x: [x, y, vx, vy, yaw]
    % 输出:
    %   z_pred: [tof_right; tof_front; tof_left]
    %   H: 3x5 雅可比矩阵

    sensor_angles = [pi/2, 0, -pi/2];  % 右、前、左
    wall_vals = [1.25, -1.15, 1.25, -1.15];  % vertical, horizontal walls

    z_pred = zeros(3,1);
    H = zeros(3,5);
    delta = 1e-5;
    idx = [1, 2, 5];  % 对 x, y, yaw 求导

    for i = 1:3
        theta = x(5);
        beta = theta + sensor_angles(i);
        pos = x(1:2);

        % 原始观测值计算
        t_candidates = [];
        if abs(cos(beta)) > 1e-5
            for c = wall_vals(1:2)
                t = (c - pos(1)) / cos(beta)
                if t > 0
                    t_candidates(end+1) = t;
                end
            end
        end
        if abs(sin(beta)) > 1e-5
            for c = wall_vals(3:4)
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
        z_pred(i) = z;

        % 数值雅可比
        grad = zeros(1,3);
        for j = 1:3
            x_pert = x;
            x_pert(idx(j)) = x_pert(idx(j)) + delta;
            pos_p = x_pert(1:2);
            beta_p = x_pert(5) + sensor_angles(i);

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
            grad(j) = (zp - z) / delta;
        end

        H(i, idx) = grad;
    end
end
