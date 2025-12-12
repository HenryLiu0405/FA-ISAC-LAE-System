function main()
    % 固定随机数种子
    rng(42);
    
    % 系统参数
    K = 3;      % 基站数量
    N = 2;      % 用户数量
    M = 4;      % 天线数量
    L = 12;     % 多径数量
    Q = 2;      % 无人机数量
    lambda = 0.1; 
    sigma_k_2 = 50; 
    theta_k = pi/2; 
    phi_k = -pi/6;
    rho = 100; 
    alpha = 0.3; % 通信感知加权因子
    P_max = 1;  % 功率约束
    
    % 优化参数
    max_alternations = 50;   % 最大交替次数
    tolerance = 1e-4;        % 收敛阈值
    
    % 初始化天线位置
    lower_bound = -0.1; 
    upper_bound = 0.1;
    tilde_t = initialize_antenna_positions(M, lower_bound, upper_bound, lambda);
    
    % 初始化波束成形矩阵
    [wc, ws] = initialize_beamforming(M, K, P_max);
    
    % 存储优化过程
    best_scores = zeros(max_alternations*2, 1);
    alt_idx = 1;
    prev_score = -inf;
    
    % 新增：存储每次交替后的性能指标
    max_alt = max_alternations;
    combined_score_history = zeros(max_alt + 1, 1);  % +1 for initial value
    comm_sinr_history = zeros(max_alt + 1, 1);
    sens_sinr_history = zeros(max_alt + 1, 1);
    antenna_pos_history = cell(max_alt + 1, 1);
    wc_history = cell(max_alt + 1, 1);
    ws_history = cell(max_alt + 1, 1);
    
    % 初始化第0次的值
    combined_score_history(1) = 0;  % Index 1对应第0次迭代
    comm_sinr_history(1) = 0;
    sens_sinr_history(1) = 0;
    antenna_pos_history{1} = tilde_t;
    wc_history{1} = wc;
    ws_history{1} = ws;
    
    window_size = 3;
    score_window = [];
    window_idx = 1;

% 交替优化主循环
for alt_iter = 1:max_alternations
    fprintf('\n=== Alternation %d ===\n', alt_iter);
    
    % --- 阶段1: 固定天线位置，优化波束成形矩阵(wc, ws) ---
    [h_k, H_ki] = compute_channels(reshape(tilde_t,2,M), M, L, Q, N, lambda, sigma_k_2, phi_k, rho);
    [wc_new, ws_new, sca_score] = sca_optimization(...
        h_k, H_ki, K, N, M, sigma_k_2, alpha, P_max);
    
    % --- 阶段2: 固定波束成形矩阵，优化天线位置 ---
    [tilde_t_new, pso_score] = pso_optimization(...
        wc_new, ws_new, M, L, Q, lambda, sigma_k_2, phi_k, rho, alpha, N, K, P_max);
    
    % ========== 更新变量 ========== %
    tilde_t = tilde_t_new;
    wc = wc_new;
    ws = ws_new;
    
    % ========== 实时分数计算 ========== %
    [h_k_current, H_ki_current] = compute_channels(reshape(tilde_t,2,M), M, L, Q, N, lambda, sigma_k_2, phi_k, rho);
    current_comm_sinr = compute_SINR_comm(h_k_current, wc, sigma_k_2, N);
    current_sens_sinr = compute_SINR_sens(H_ki_current, ws, sigma_k_2, K);
    current_combined_score = alpha*log2(1 + current_comm_sinr) + (1-alpha)*log2(1 + current_sens_sinr);
    
    % ========== 显示本次交替结果 ========== %
    fprintf('[当前分数] 通信SINR: %.2f dB | 感知SINR: %.2f dB | 综合指标: %.4f\n',...
        10*log10(current_comm_sinr),...
        10*log10(current_sens_sinr),...
        current_combined_score);
    
    % ========== 存储历史数据 ========== %
    combined_score_history(alt_iter+1) = current_combined_score;
    comm_sinr_history(alt_iter+1) = current_comm_sinr;
    sens_sinr_history(alt_iter+1) = current_sens_sinr;
    antenna_pos_history{alt_iter+1} = tilde_t;
    wc_history{alt_iter+1} = wc;
    ws_history{alt_iter+1} = ws;
    
    % ========== 动态窗口收敛检查 ========== %
    score_window = [score_window, current_combined_score]; % 添加最新得分
    if length(score_window) > window_size
        score_window(1) = []; % 维持窗口大小
    end
    
    % 检查窗口变化（需要至少3次迭代后检查）
    if length(score_window) == window_size
        avg_change = mean(abs(diff(score_window)));
        fprintf('窗口平均变化: %.6f (阈值: %.4f)\n', avg_change, tolerance);
        if avg_change < tolerance
            fprintf('----> 收敛条件满足，优化终止 <----\n');
            break;
        end
    end
    
    % ========== 传统收敛检查 ========== %
    delta_score = abs(current_combined_score - prev_score);
    fprintf('分数变化量: %.6f (阈值: %.4f)\n', delta_score, tolerance);
    if delta_score < tolerance
        fprintf('----> 收敛条件满足，优化终止 <----\n');
        break;
    end
    prev_score = current_combined_score; % 更新前次分数记录
end
    
    % 截断历史记录到实际交替次数
    num_alt = alt_iter;  % 实际交替次数
    combined_score_history = combined_score_history(1:num_alt+1);
    comm_sinr_history = comm_sinr_history(1:num_alt+1);
    sens_sinr_history = sens_sinr_history(1:num_alt+1);
    antenna_pos_history = antenna_pos_history(1:num_alt+1);
    wc_history = wc_history(1:num_alt+1);
    ws_history = ws_history(1:num_alt+1);
    
    % 输出最终结果并绘图
    plot_all_results(combined_score_history, comm_sinr_history, sens_sinr_history, ...
        wc_history, ws_history, antenna_pos_history, lambda);
    disp('Optimization Complete.');
    disp('Final Antenna Positions:'); 
    disp(reshape(tilde_t, [2, M]));
    disp('Final wc:'); disp(wc);
    disp('Final ws:'); disp(ws);
end
%% 初始化函数




function tilde_t = initialize_antenna_positions(M, lb, ub, lambda)
    min_spacing = 0.5 * lambda;
    tilde_t = zeros(2, M);
    attempts = 0;
    while true
        tilde_t = lb + (ub - lb)*rand(2, M);
        if check_spacing_constraint(tilde_t, min_spacing)
            break;
        end
        attempts = attempts + 1;
        if attempts > 1000
            error('无法生成满足间距约束的初始天线位置');
        end
    end
    tilde_t = tilde_t(:)';
end

function [wc, ws] = initialize_beamforming(M, K, P_max)
    % 随机初始化并满足功率约束
    wc = (randn(M, K) + 1j*randn(M, K))/sqrt(2);
    ws = (randn(M, K) + 1j*randn(M, K))/sqrt(2);
    
    % 总功率归一化
    total_power = norm(wc, 'fro')^2 + norm(ws, 'fro')^2;
    scaling_factor = sqrt(P_max / total_power);
    wc = wc * scaling_factor;
    ws = ws * scaling_factor;
end

%% 新增的初始化函数
function solutions = initialize_sca_solutions(num_solutions, dim, lb, ub)
    % 参数验证
    assert(mod(dim,2)==0, '维度必须是偶数');
    
    % 生成初始解矩阵（实数域）
    solutions = lb + (ub - lb) * rand(num_solutions, dim);
    
    % 功率补偿因子（防止初始解过小）
    power_boost = 10;  % 根据实际情况调整
    
    for i = 1:num_solutions
        real_part = solutions(i, 1:dim/2);
        imag_part = solutions(i, dim/2+1:end);
        complex_sol = complex(real_part, imag_part);
        solutions(i, :) = [real(complex_sol*power_boost), imag(complex_sol*power_boost)];
    end
end

function particles = initialize_particles(num_particles, dim, lb, ub, lambda)
    % 参数验证
    assert(dim > 0, '维度必须为正整数');
    assert(ub > lb, '上界必须大于下界');
    
    % 生成均匀分布的粒子位置（实数域）
    particles = lb + (ub - lb)*rand(num_particles, dim);
    for i = 1:num_particles
        while true
            pos = reshape(particles(i,:), 2, []);
            if check_spacing_constraint(pos, lambda*0.7)
                break;
            end
            particles(i,:) = lb + (ub - lb) * rand(1, dim);
        end
    end
end

function valid = check_spacing_constraint(antenna_pos, min_spacing)
    % antenna_pos: 2xM矩阵
    % 检查所有天线间距是否满足最小间距要求
    valid = true;
    M = size(antenna_pos, 2);
    for i = 1:M-1
        for j = i+1:M
            dx = antenna_pos(1,i) - antenna_pos(1,j);
            dy = antenna_pos(2,i) - antenna_pos(2,j);
            d = sqrt(dx^2 + dy^2);
            if d < min_spacing
                valid = false;
                return;
            end
        end
    end
end

%% SCA优化模块
function [wc_opt, ws_opt, best_score] = sca_optimization(...
    h_k, H_ki, K, N, M, sigma_k_2, alpha, P_max)
    
    % SCA参数
    num_solutions = 100; 
    max_iter = 1000; 
    dim = 2*M*K; 
    lower_bound = -1; 
    upper_bound = 1;
    
    % 初始化解
    solutions = initialize_sca_solutions(num_solutions, dim, lower_bound, upper_bound);
    best_solution = solutions(1, :); 
    best_score = -inf;
    
    for iter = 1:max_iter
        for i = 1:num_solutions
            % 提取波束成形矩阵
            wc = reshape(solutions(i, 1:M*K), [M, K]);
            ws = reshape(solutions(i, M*K+1:end), [M, K]);
            
            % 新增：功率约束处理
            total_power = norm(wc, 'fro')^2 + norm(ws, 'fro')^2;
            if total_power > P_max
                % 功率归一化
                scaling_factor = sqrt(P_max / total_power);
                wc = wc * scaling_factor;
                ws = ws * scaling_factor;
                solutions(i, :) = [wc(:); ws(:)]'; % 更新解
            end
            
            % 计算目标函数（新增功率惩罚项）
            SINR_comm = compute_SINR_comm(h_k, wc, sigma_k_2, N);
            SINR_sens = compute_SINR_sens(H_ki, ws, sigma_k_2, K);
            power_penalty = max(0, (norm(wc, 'fro')^2 + norm(ws, 'fro')^2)/P_max - 1);
            score = alpha*log2(1 + SINR_comm) + (1-alpha)*log2(1 + SINR_sens) - 1e3*power_penalty;
            
            % 更新最优解
            if score > best_score
                best_score = score;
                best_solution = solutions(i, :);
            end
        end
        
        % SCA更新规则
        solutions = sca_update(solutions, best_solution, dim, ...
            lower_bound, upper_bound, max_iter, iter);
    end
    
    % 提取最优解
    wc_opt = reshape(best_solution(1:M*K), [M, K]);
    ws_opt = reshape(best_solution(M*K+1:end), [M, K]);
end

function solutions = sca_update(solutions, best_solution, dim, lb, ub, max_iter, iter)
    % 非线性递减参数
    r1 = 2*(1 - (iter/max_iter))^2; 
    w = 0.7 - 0.3*(iter/max_iter);
    
    for i = 1:size(solutions,1)
        for j = 1:dim
            r2 = 2*pi*rand();
            r3 = 0.5 + rand();
            r4 = rand();
            
            if r4 < 0.5
                solutions(i,j) = w*solutions(i,j) + r1*sin(r2)*abs(r3*best_solution(j) - solutions(i,j));
            else
                solutions(i,j) = w*solutions(i,j) + r1*cos(r2)*abs(r3*best_solution(j) - solutions(i,j));
            end
        end
        % 边界处理
        solutions(i,:) = max(min(solutions(i,:), ub), lb);
        solutions(i,:) = solutions(i,:)/norm(solutions(i,:)); % 功率归一化
    end
end

%% PSO优化模块
function [tilde_t_opt, best_score] = pso_optimization(...
    wc, ws, M, L, Q, lambda, sigma_k_2, phi_k, rho, alpha, N, K, P_max)
    
    % PSO参数
    num_particles = 100; 
    max_iter = 1000; 
    dim = 2*M; 
    lb = -0.1; 
    ub = 0.1;
    w = 0.7; 
    c1 = 1.5; 
    c2 = 1.5;
    
    % 初始化粒子群
    particles = initialize_particles(num_particles, dim, lb, ub, lambda);
    velocities = zeros(num_particles, dim);
    personal_best = particles; 
    personal_scores = -inf(num_particles,1);
    global_best = particles(1,:); 
    global_score = -inf;
    
    for iter = 1:max_iter
        for i = 1:num_particles
            % 计算目标函数
            score = pso_objective(particles(i,:), wc, ws, ...
                M, L, Q, lambda, sigma_k_2, phi_k, rho, alpha, N, K, P_max);
            
            % 更新个体最优
            if score > personal_scores(i)
                personal_scores(i) = score;
                personal_best(i,:) = particles(i,:);
            end
            
            % 更新全局最优
            if score > global_score
                global_score = score;
                global_best = particles(i,:);
            end
        end
        
        % 更新粒子速度和位置
        for i = 1:num_particles
            velocities(i,:) = w*velocities(i,:) + ...
                c1*rand()*(personal_best(i,:) - particles(i,:)) + ...
                c2*rand()*(global_best - particles(i,:));
            
            particles(i,:) = particles(i,:) + velocities(i,:);
            % 边界处理
            particles(i,:) = max(min(particles(i,:), ub), lb);
        end
    end
    
    tilde_t_opt = global_best;
    best_score = global_score;
end

function score = pso_objective(tilde_t, wc, ws, M, L, Q, lambda, sigma_k_2, phi_k, rho, alpha, N, K, P_max)
    % 功率约束
    total_power = norm(wc, 'fro')^2 + norm(ws, 'fro')^2;
    if total_power > P_max
        score = -inf;
        return;
    end
    
    % 提取天线位置并检查间距
    antenna_pos = reshape(tilde_t, 2, M);
    min_spacing = 0.7 * lambda; % 最小间距
    if ~check_spacing_constraint(antenna_pos, min_spacing)
        score = -inf;
        return;
    end
    
    % 计算信道和SINR
    [h_k, H_ki] = compute_channels(antenna_pos, M, L, Q, N, lambda, sigma_k_2, phi_k, rho);
    SINR_comm = compute_SINR_comm(h_k, wc, sigma_k_2, N);
    SINR_sens = compute_SINR_sens(H_ki, ws, sigma_k_2, K);
    score = alpha*log2(1 + SINR_comm) + (1-alpha)*log2(1 + SINR_sens);
end

%% 公共函数

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

function SINR_comm = compute_SINR_comm(h_k, wc, sigma_k_2, N)
    % 计算通信SINR（使用迹运算）
    Wc1 = wc(:,1) * wc(:,1)';
    hk_hk = h_k * h_k';
    signal = trace(Wc1 * hk_hk);
    
    interference = 0;
    for n = 1:N
        Wcn = wc(:,n) * wc(:,n)';
        interference = interference + trace(Wcn * hk_hk);
    end
    SINR_comm = signal / (interference - signal + sigma_k_2);
end

function SINR_sens = compute_SINR_sens(H_ki, ws, sigma_k_2, K)
    % 计算感知SINR（使用迹运算）
    Ws1 = ws(:,1) * ws(:,1)';
    H_ki_H = H_ki * H_ki';  % 计算H_ki的Hermitian乘积
    signal = trace(Ws1 * H_ki_H);
    
    interference = 0;
    for k = 1:K
        Wsk = ws(:,k) * ws(:,k)';
        interference = interference + trace(Wsk * H_ki_H);
    end
    SINR_sens = signal / (interference - signal + sigma_k_2);
end

function plot_all_results(combined_scores, comm_sinr, sens_sinr, ...
    wc_history, ws_history, antenna_pos_history, lambda)
    
   % 生成正确的x轴坐标
    iterations = 0:(length(combined_scores)-1);
    
    % 转换为dB显示
    comm_sinr_db = 10*log10(comm_sinr);
    sens_sinr_db = 10*log10(sens_sinr);
    
    %% 综合评分收敛曲线（带初始点）
    figure('Name','Convergence','Position',[100 100 800 400])
    plot(iterations, combined_scores, 'LineWidth', 2, 'Color',[0.2 0.6 0.8],...
        'Marker','o','MarkerSize',6)
    hold on
    plot(0, combined_scores(1), 'ro','MarkerSize',8,'LineWidth',2) % 突出显示初始点
    xlabel('Alternation Index');
    ylabel('Combined Metric');
    title('Alternating Optimization Convergence');
    grid on;
    set(gca,'FontSize',12)
    xlim([-0.5 max(iterations)+0.5])
    
    %% 通信与感知SINR曲线（带初始点）
    figure('Name','SINR Performance','Position',[100 100 800 600])
    
    subplot(2,1,1)
    plot(iterations, comm_sinr_db, 'LineWidth',2, 'Color',[0 0.4 0.8],...
        'Marker','s','MarkerSize',6)
    hold on
    plot(0, comm_sinr_db(1), 'ro','MarkerSize',8,'LineWidth',2)
    title('Communication SINR');
    xlabel('Alternation Index');
    ylabel('SINR (dB)');
    grid on;
    set(gca,'FontSize',12)
    xlim([-0.5 max(iterations)+0.5])
    
    subplot(2,1,2)
    plot(iterations, sens_sinr_db, 'LineWidth',2, 'Color',[0.8 0.2 0.2],...
        'Marker','d','MarkerSize',6)
    hold on
    plot(0, sens_sinr_db(1), 'ro','MarkerSize',8,'LineWidth',2)
    title('Sensing SINR');
    xlabel('Alternation Index');
    ylabel('SINR (dB)');
    grid on;
    set(gca,'FontSize',12)
    xlim([-0.5 max(iterations)+0.5])
    
    % 波束成形权重图（最后一次迭代结果）
    % 绘制 wc 和 ws 的分布
    figure;
    subplot(1, 2, 1);
    imagesc(wc_history{end});
    title('Optimized wc');
    xlabel('User Index');
    ylabel('Antenna Index');
    colorbar;

    subplot(1, 2, 2);
    imagesc(ws_history{end});
    title('Optimized ws');
    xlabel('User Index');
    ylabel('Antenna Index');
    colorbar;
    
    % 天线位置迭代图
    figure;
    hold on;
    colors = jet(length(antenna_pos_history));
    for i = 1:length(antenna_pos_history)
        pos = reshape(antenna_pos_history{i}, 2, []);
        scatter(pos(1,:), pos(2,:), 50, colors(i,:), 'filled', ...
            'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
    end
    xlabel('X Position (m)');
    ylabel('Y Position (m)');
    title('Antenna Positions Evolution');
    colorbar('Ticks', linspace(0,1,length(antenna_pos_history)), ...
        'TickLabels', arrayfun(@(x) sprintf('Iter %d',x), 1:length(antenna_pos_history), 'UniformOutput', false));
    grid on;
    hold off;
    
    % 最终天线位置图（带间距检查）
    final_pos = reshape(antenna_pos_history{end}, 2, []);
    min_spacing = lambda * 0.5; % 最小间距要求
    
    figure;
    scatter(final_pos(1,:), final_pos(2,:), 100, 'b', 'filled');
    hold on;
    % 绘制天线间连接线以显示间距
    for i = 1:size(final_pos,2)
        for j = i+1:size(final_pos,2)
            line([final_pos(1,i), final_pos(1,j)], [final_pos(2,i), final_pos(2,j)], ...
                'Color', [0.5 0.5 0.5], 'LineStyle', '--');
        end
    end
    % 标注间距
    for i = 1:size(final_pos,2)
        text(final_pos(1,i)+0.005, final_pos(2,i), sprintf('Ant %d',i), ...
            'FontSize', 8, 'HorizontalAlignment', 'left');
    end
    title(sprintf('Final Antenna Positions\n(Min Spacing: %.2fλ)', min_spacing));
    xlabel('X Position (m)');
    ylabel('Y Position (m)');
    axis equal;
    grid on;
    
    % 检查间距约束
    if check_spacing_constraint(final_pos, min_spacing)
        disp('Antenna spacing constraint satisfied.');
    else
        warning('Antenna spacing constraint violated!');
    end
end