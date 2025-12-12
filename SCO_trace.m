function main
    % 固定随机数种子
    rng(42);

    % 参数设置
    K = 3;  N = 2;  M =4;  L = 12;  Q = 1;  
    lambda = 0.1; sigma_k_2 = 50; theta_k = pi/2;  
    phi_k = -pi/6; rho = 100; alpha = 0.7; 

    % SCA 参数
    num_solutions = 50; max_iter = 500;  
    dim = 2 * M * K; lower_bound = -1; upper_bound = 1;

    % 初始化解
    solutions = initialize_solutions(num_solutions, dim, lower_bound, upper_bound);
    best_solution = solutions(1, :); best_score = -inf; 
    best_scores = zeros(max_iter, 1);

    % 主循环
    for iter = 1:max_iter
        for i = 1:num_solutions
            wc = reshape(solutions(i, 1:M*K), [M, K]);
            ws = reshape(solutions(i, M*K+1:end), [M, K]);
            
            [h_k, H_ki] = compute_channels(M, L, Q, N, lambda, sigma_k_2, phi_k, rho);
            [current_score, current_comm, current_sens] = objective_function(wc, ws, h_k, H_ki, sigma_k_2, alpha, K, N);
            
            if current_score > best_score
                best_score = current_score;
                best_solution = solutions(i, :);
                best_comm = current_comm; % 更新最佳通信SINR
                best_sens = current_sens; % 更新最佳感知SINR
            end
        end
        
        solutions = update_solutions(solutions, best_solution, dim, lower_bound, upper_bound, max_iter, iter);
        best_scores(iter) = best_score;
        best_comm_scores(iter) = best_comm; % 记录当前迭代的最佳通信SINR
        best_sens_scores(iter) = best_sens; % 记录当前迭代的最佳感知SINR;
        fprintf('Iteration %d, Best Score: %.4f\n', iter, best_score);
    end

    % 输出结果
    wc_opt = reshape(best_solution(1:M*K), [M, K]);
    ws_opt = reshape(best_solution(M*K+1:end), [M, K]);
    disp('Optimized wc and ws:'); disp(wc_opt); disp(ws_opt);
    disp(['Best Weighted SINR: ', num2str(best_score)]);
    
    plot_results(best_scores, best_comm_scores, best_sens_scores, wc_opt, ws_opt);
end

function solutions = initialize_solutions(num_solutions, dim, lower_bound, upper_bound)
    % 初始化解
    solutions = lower_bound + (upper_bound - lower_bound) * rand(num_solutions, dim);
    % 归一化波束成形矩阵
    for i = 1:num_solutions
        solutions(i, :) = solutions(i, :) / norm(solutions(i, :));
    end
end

function solutions = update_solutions(solutions, best_solution, dim, lower_bound, upper_bound, max_iter, iter)
    % 更新解
    r1 = 2 * (1 - (iter / max_iter))^2; % 非线性递减的 r1
    w = 0.7 - 0.3 * (iter / max_iter); % 惯性权重

    for i = 1:size(solutions, 1)
        for j = 1:dim
            r2 = 2 * pi * rand(); % 随机参数
            r3 = 0.5 + rand() * 1.0; % 调整 r3 的范围
            r4 = rand(); % 随机参数

            if r4 < 0.5
                % 使用正弦函数更新解
                solutions(i, j) = w * solutions(i, j) + r1 * sin(r2) * abs(r3 * best_solution(j) - solutions(i, j));
            else
                % 使用余弦函数更新解
                solutions(i, j) = w * solutions(i, j) + r1 * cos(r2) * abs(r3 * best_solution(j) - solutions(i, j));
            end
        end

        % 边界处理
        solutions(i, :) = max(solutions(i, :), lower_bound);
        solutions(i, :) = min(solutions(i, :), upper_bound);

        % 归一化波束成形矩阵
        solutions(i, :) = solutions(i, :) / norm(solutions(i, :));
    end
end

function [score, SINR_comm, SINR_sens] = objective_function(wc, ws, h_k, H_ki, sigma_k_2, alpha, K, N)
    % 通信SINR计算（迹形式）
    SINR_comm = compute_SINR_comm(h_k, wc, sigma_k_2, N);
    
    % 感知SINR计算（迹形式）
    SINR_sens = compute_SINR_sens(H_ki, ws, sigma_k_2, K);
    
    % 加权SINR
    score = alpha * log2(1 + max(SINR_comm, eps)) + (1 - alpha) * log2(1 + max(SINR_sens, eps));
end

function [h_k, H_ki] = compute_channels(M, L, Q, N, lambda, sigma_k_2, phi_k, rho)
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
    h_k_hkH = h_k * h_k';
    wc_wcH = wc * wc';
    sum_term = trace(wc_wcH * h_k_hkH); % 总干扰项
    
    SINR_comm = zeros(N, 1);
    for n = 1:N
        wc_n = wc(:, n);
        numerator = trace(wc_n' * h_k_hkH * wc_n); % 取实部避免数值误差
        denominator = sum_term - numerator + sigma_k_2;
        SINR_comm(n) = numerator / denominator;
    end
    
    SINR_comm = mean(SINR_comm);
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

function plot_results(best_scores, best_comm, best_sens, wc_opt, ws_opt)
    % 绘制 SINR 收敛曲线
    figure;
    plot(1:length(best_scores), best_scores, 'LineWidth', 2);
    title('Weighted SINR Convergence');
    xlabel('Iteration');
    ylabel('Weighted SINR');
    grid on;

    figure;
    plot(1:length(best_comm),best_comm,'--','LineWidth',2)
    title('communication SINR convergence');
    xlabel('Iteration');
    ylabel('communication SINR');
    grid on;

    figure;
    plot(1:length(best_sens),best_sens,':','LineWidth',2)
    title('sensing SINR convergence');
    xlabel('Iteration');
    ylabel('sensing SINR');
    grid on;

    % 绘制 wc 和 ws 的分布
    figure;
    subplot(1, 2, 1);
    imagesc(wc_opt);
    title('Optimized wc');
    xlabel('User Index');
    ylabel('Antenna Index');
    colorbar;

    subplot(1, 2, 2);
    imagesc(ws_opt);
    title('Optimized ws');
    xlabel('User Index');
    ylabel('Antenna Index');
    colorbar;
end