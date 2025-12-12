function main
    % 固定随机数种子
    rng(42);

    % 参数设置
    K = 3; % 基站数量
    N = 2; % 用户数量
    M = 4; % 阵列天线数量
    L = 12; % 路径数
    Q = 100; % 无人机个数
    lambda = 0.1; % 波长
    sigma_k_2 = 50; % 噪声功率
    theta_k = pi/2; % 仰角
    phi_k = -pi/6; % 方位角
    rho = -100; % 路径差
    alpha = 0.3; % 加权因子
    P_max = 1;

    % PSO 参数
    num_particles = 50; % 粒子数量
    max_iter = 500; % 最大迭代次数
    w = 0.7; % 惯性权重
    c1 = 1.5; % 个体学习因子
    c2 = 1.5; % 社会学习因子
    dim = 2 * M; % 优化变量维度
    lower_bound = -0.1; % FA 位置下界
    upper_bound = 0.1; % FA 位置上界

    % 初始化粒子群
    particles = initialize_particles(num_particles, dim, lower_bound, upper_bound);
    velocities = zeros(num_particles, dim); % 初始化速度
    personal_best_positions = particles; % 个体最优位置
    personal_best_scores = -inf(num_particles, 1); % 个体最优得分
    global_best_position = particles(1, :); % 全局最优位置
    global_best_score = -inf; % 全局最优得分

    % 用于存储每次迭代的全局最优得分
    global_best_scores = zeros(max_iter, 1);
    comm_sinr = zeros(max_iter, 1);
    sens_sinr = zeros(max_iter, 1);

    % 主循环
    for iter = 1:max_iter
        for i = 1:num_particles
            % 提取当前粒子的 FA 位置
            tilde_t = particles(i, :);

            % 计算目标函数值
            current_score = objective_function(tilde_t, K, N, M, L, Q, lambda, sigma_k_2, theta_k, phi_k, rho, alpha, P_max);

            % 更新个体最优
            if current_score > personal_best_scores(i)
                personal_best_scores(i) = current_score;
                personal_best_positions(i, :) = tilde_t;
            end

            % 更新全局最优
            if current_score > global_best_score
                global_best_score = current_score;
                global_best_position = tilde_t;
            end
        end

        % 记录当前迭代的全局最优得分
        global_best_scores(iter) = global_best_score;
        % 在迭代结束后计算当前最优解的SINR值（新增部分）
        [~, current_comm, current_sens] = objective_function(global_best_position, K, N, M, L, Q, lambda, sigma_k_2, theta_k, phi_k, rho, alpha, P_max);
        comm_sinr(iter) = current_comm;
        sens_sinr(iter) = current_sens;

        % 更新粒子速度和位置
        for i = 1:num_particles
            r1 = rand(1, dim);
            r2 = rand(1, dim);
            velocities(i, :) = w * velocities(i, :) + ...
                c1 * r1 .* (personal_best_positions(i, :) - particles(i, :)) + ...
                c2 * r2 .* (global_best_position - particles(i, :));
            particles(i, :) = particles(i, :) + velocities(i, :);

            % 边界处理
            particles(i, :) = max(particles(i, :), lower_bound);
            particles(i, :) = min(particles(i, :), upper_bound);
        end

        % 显示当前迭代结果
        fprintf('Iteration %d, Best Score: %.4f\n', iter, global_best_score);
    end

    % 输出最终结果
    disp('Optimized FA Positions:');
    disp(reshape(global_best_position, [2, M]));
    disp(['Best Weighted SINR: ', num2str(global_best_score)]);

    % 绘制图像
    plot_results(global_best_scores, reshape(global_best_position, [2, M]), comm_sinr, sens_sinr);
end


function particles = initialize_particles(num_particles, dim, lower_bound, upper_bound)
    % 初始化粒子群
    particles = lower_bound + (upper_bound - lower_bound) * rand(num_particles, dim);
end

function [score, comm_sinr_val, sens_sinr_val] = objective_function(tilde_t, K, N, M, L, Q, lambda, sigma_k_2, theta_k, phi_k, rho, alpha, P_max)
    % 1. 将粒子位置 tilde_t 转换为天线位置
    antenna_positions = reshape(tilde_t, [2, M]); % 假设 tilde_t 是 2xM 的矩阵，表示天线的 x 和 y 坐标

    % 检查天线间距约束（新增部分）
    d_min = lambda / 2; % 最小间距设为半波长
    violation = false;
    for i = 1:M-1
        for j = i+1:M
            dx = antenna_positions(1, i) - antenna_positions(1, j);
            dy = antenna_positions(2, i) - antenna_positions(2, j);
            d = sqrt(dx^2 + dy^2);
            if d < d_min
                violation = true;
                break;
            end
        end
        if violation
            break;
        end
    end
    if violation
        score = -inf; % 违反约束，返回负无穷
        return;
    end

    % 2. 计算信道向量 h_k 和信道矩阵 H_ki
    [h_k, H_ki] = compute_channels(antenna_positions, M, L, Q, N, lambda, sigma_k_2, phi_k, rho);

    % 3. 动态生成波束成形矩阵
    [wc, ws] = generate_beamforming_matrices(antenna_positions, M, K, lambda, theta_k, phi_k, P_max);

    % 4. 计算通信 SINR
    SINR_comm = compute_SINR_comm(h_k, wc, sigma_k_2);

    % 5. 计算感知 SINR
    SINR_sens = compute_SINR_sens(H_ki, ws, sigma_k_2, K);

    % 6. 计算加权 SINR
    score = alpha * log2(1 + SINR_comm) + (1 - alpha) * log2(1 + SINR_sens);
    comm_sinr_val = SINR_comm;
    sens_sinr_val = SINR_sens;
end

function [wc, ws] = generate_beamforming_matrices(antenna_positions, M, K, lambda, theta_k, phi_k, P_max)
    % 生成导向矢量
    wc = zeros(M, K);
    for k = 1:K
        x_coords = antenna_positions(1,:);
        y_coords = antenna_positions(2,:);
        phase = 2*pi/lambda * (x_coords*sin(theta_k)*cos(phi_k) + y_coords*sin(theta_k)*sin(phi_k));
        wc(:,k) = exp(1j*phase);
    end
    
    % 功率归一化
    total_power = norm(wc,'fro')^2;
    scaling = sqrt(P_max / total_power);
    wc = wc * scaling;
    ws = wc; % 简化的感知波束
end

function [h_k, H_ki] = compute_channels(antenna_positions, M, L, Q, N, lambda, sigma_k_2, phi_k, rho)
    % 初始化场响应矩阵
    G_kq = zeros(L, M); % 大小为 L x M
    G_qi = zeros(L, Q); % 大小为 L x M
    G_kn = zeros(L, M);

    % 初始化 H
    H_kq = zeros(M, Q);
    H_qi = zeros(Q, M);

     % 计算场响应矩阵
    for m = 1:M
        G_kq(:, m) = exp(1j * 2 * pi / lambda * (rho + m * cos(phi_k) + m * sin(phi_k))).';
    end
    for q = 1:Q
        G_qi(:, q) = exp(1j * 2 * pi / lambda * (rho + q * cos(phi_k) + q * sin(phi_k))).';
    end
    for n = 1:N
        G_kn(:, n) = exp(1j * 2 * pi / lambda * (rho + n * cos(phi_k) + n * sin(phi_k))).';
    end

    % 信道向量 h_k(tilde_t)
    Sigma_k = sigma_k_2 * ones(L, 1); % 路径响应矩阵
    H_kq = G_kq' * Sigma_k;
    H_qi = G_qi' * Sigma_k;

    % 计算 H_ki
    H_ki = 1e-3 * H_kq * H_qi';

    % 返回 h_k 和 H_ki
    h_k = G_kn' * Sigma_k; % 假设 h_k 为 h_kq
end

function SINR_comm = compute_SINR_comm(h_k, wc, sigma_k_2)
    [M, N] = size(h_k);
    SINR_comm = zeros(N,1);
    for n = 1:N
        signal = abs(wc(:,n)'*h_k(:,n))^2;
        interference = sum(abs(wc(:,setdiff(1:N,n))'*h_k(:,n)).^2);
        SINR_comm(n) = signal/(interference + sigma_k_2);
    end
    SINR_comm = mean(SINR_comm); % 转换为dB
end

function SINR_sens = compute_SINR_sens(H_ki, ws, sigma_k_2, K)
    H_ki_HkiH = H_ki * H_ki';
    ws_wsH = ws * ws';
    sum_term = trace(ws_wsH * H_ki_HkiH); % 总干扰项

    SINR_sens = zeros(K, 1);
    for k = 1:K
        ws_k = ws(:, k);
        numerator = trace(ws_k' * H_ki_HkiH * ws_k); % 取实部避免数值误差
        denominator = sum_term - numerator + sigma_k_2;
        SINR_sens(k) = numerator / denominator;
    end
    SINR_sens = mean(SINR_sens);
end

function plot_results(global_best_scores, fa_positions, comm_sinr, sens_sinr)
    % 绘制 SINR 收敛曲线
    figure;
    plot(1:length(global_best_scores), global_best_scores, 'LineWidth', 2);
    title('Weighted SINR Convergence');
    xlabel('Iteration');
    ylabel('Weighted SINR');
    grid on;

    % 绘制 FA 位置
    figure;
    scatter(fa_positions(1, :), fa_positions(2, :), 100, 'filled');
    title('Optimized FA Positions');
    xlabel('X Coordinate');
    ylabel('Y Coordinate');
    grid on;
    axis([-0.1 0.1 -0.1 0.1]);

    figure;
    subplot(2,1,1);
    plot(1:length(comm_sinr), comm_sinr, 'b', 'LineWidth', 2);
    xlabel('Iteration');
    ylabel('SINR (dB)');
    title('Communication SINR Convergence');
    grid on;

    subplot(2,1,2);
    plot(1:length(sens_sinr), sens_sinr, 'r', 'LineWidth', 2);
    xlabel('Iteration');
    ylabel('SINR (dB)');
    title('Sensing SINR Convergence');
    grid on;
end