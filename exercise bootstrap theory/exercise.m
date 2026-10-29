% Exercise - Understanding the block bootstrap using simulation

%% 1. Aim of the exercise

% This exercise illustrates when the block bootstrap approximates the
% sampling distribution of a statistic computed from serially dependent
% data. The data come from a stationary AR(1) process, and the statistic is
% the scaled sample mean. The bootstrap resamples blocks of consecutive
% observations, and its accuracy depends on the block length l. Consistency
% requires that l grows with the sample size n (l -> infinity), but more
% slowly than n (l/n -> 0). Three block-length rules are compared: l = 1,
% which violates the first condition and reduces to the standard bootstrap,
% l = n^(1/3), which satisfies both conditions, and l = n/5, which violates
% the second condition.

%% 2. Initialize MATLAB environment

% Clear workspace, command window, and open figures
clear;
clc;
close all;

%% 3. Construct the data-generating process and analytical distributions

% 3.1. Set a seed for the random number generator for reproducibility
rng(26)

% 3.2. Set the autoregressive parameter of the AR(1) process
phi = 0.7;

% 3.3. Set the standard deviation of the innovations
sigma_eps = 1;

% 3.4. Compute the variance of the AR(1) process
gamma_0 = sigma_eps^2 / (1 - phi^2);

% 3.5. Define a function that simulates a stationary AR(1) series from a
% vector of innovations. The first observation is drawn from the
% stationary distribution, so that the series is stationary from the
% start.
ar1 = @(epsilon) filter(1,[1 -phi], ...
    [epsilon(1) / sqrt(1 - phi^2); epsilon(2:end)]);

% 3.6. Define the variance of the statistic for sample size n
v_pop = @(n) gamma_0 * (1 + 2 * sum((1 - (1:n-1) / n) .* phi.^(1:n-1)));

% 3.7. Define the sampling PDF of the statistic
g_pop = @(tau,n) pdf('Normal',tau,0,sqrt(v_pop(n)));

% 3.8. Define the sampling CDF of the statistic
G_pop = @(tau,n) cdf('Normal',tau,0,sqrt(v_pop(n)));

% 3.9. Set the grid of tau values at which PDFs and CDFs are evaluated
tau_grid = linspace(-15,15,1000)';

% 3.10. Set the sample sizes
N_obs_vec = [125 1000 8000];

% 3.11. Draw a sequence of observations from the AR(1) process
data_all = ar1(random('Normal',0,sigma_eps,[max(N_obs_vec),1]));

% The statistic of interest is Tn = sqrt(n) * (sample mean - mu), with mu =
% 0. Because the AR(1) process is Gaussian, the sample mean is normally
% distributed, and the sampling distribution of Tn is known analytically:
% Tn ~ N(0, v_n). This allows the bootstrap approximation to be compared
% directly with the true sampling distribution. The sample sizes are chosen
% so that n^(1/3) and n/5 are integers.

%% 4. Set the block-length rules

% 4.1. Define the block-length rules as functions of the sample size
block_rule = {@(n) 1, @(n) round(n^(1/3)), @(n) n / 5};

% 4.2. Set the labels of the block-length rules
block_label = {'l = 1', 'l = n^{1/3}', 'l = n/5'};

% 4.3. Set the number of block-length rules
N_rules = length(block_rule);

% 4.4. Set the sample sizes
N_n = length(N_obs_vec);

%% 5. Construct block bootstrap distributions for different n and l

% 5.1. Set the number of bootstrap replications
N_boot = 10^4;

% 5.2. Preallocate storage for the bootstrap statistic
T_boot = NaN(N_boot,N_rules,N_n);

% 5.3. For each block-length rule and sample size, select the original
% sample, divide it into non-overlapping blocks, draw bootstrap samples of
% blocks, and compute the bootstrap statistic
for j = 1:N_rules
    for i = 1:N_n
        n = N_obs_vec(i);
        l = block_rule{j}(n);
        k = n / l;
        data_sample = data_all(1:n);
        % Each row of blocks is one block of l consecutive observations
        blocks = reshape(data_sample,l,k)';
        for b = 1:N_boot
            data_boot = datasample(blocks,k,1,'Replace',true);
            % Tn* = sqrt(n) * (bootstrap sample mean - 
            % original sample mean)
            T_boot(b,j,i) = sqrt(n) * ...
                (mean(data_boot(:)) - mean(data_sample));
        end
    end
end

%% 6. Convergence of the block bootstrap PDF to the sampling PDF

% Plot the block bootstrap PDF and the true sampling PDF for each
% block-length rule (rows) and sample size (columns).
figure('Position',[100 100 1000 1000]);
layout = tiledlayout(N_rules,N_n, ...
    'TileSpacing','compact','Padding','compact');
for j = 1:N_rules
    for i = 1:N_n
        n = N_obs_vec(i);
        l = block_rule{j}(n);
        nexttile
        hold on
        plot(tau_grid,ksdensity(T_boot(:,j,i),tau_grid,'function','pdf'))
        plot(tau_grid,g_pop(tau_grid,n))
        title(['n = ' num2str(n) ',l = ' num2str(l)])
        xlim([tau_grid(1) tau_grid(end)])
        if i == 1
            ylabel(block_label{j})
        end
        hold off
    end
end
x = xlabel(layout,'\tau');
y = ylabel(layout,'Density');
x.FontSize = get(groot,'defaultAxesFontSize');
y.FontSize = get(groot,'defaultAxesFontSize');
lgd = legend( ...
    'Block bootstrap PDF', ...
    'Sampling PDF', ...
    'Orientation','horizontal');
lgd.Layout.Tile = 'south';
t = title(layout,['Fig. 1. Block bootstrap PDF and sampling PDF ' ...
    'as n grows (B = ' num2str(N_boot) ')']);
t.FontSize = get(groot,'defaultAxesFontSize');
t.FontWeight = 'normal';

%% 7. Convergence of the block bootstrap CDF to the sampling CDF

% Plot the block bootstrap CDF and the true sampling CDF for each
% block-length rule (rows) and sample size (columns). The title of each
% panel reports the largest distance between the two CDFs over the grid of
% tau values.
dist_max = NaN(N_rules,N_n);
figure('Position',[100 100 1000 1000]);
layout = tiledlayout(N_rules,N_n, ...
    'TileSpacing','compact','Padding','compact');
for j = 1:N_rules
    for i = 1:N_n
        n = N_obs_vec(i);
        l = block_rule{j}(n);
        G_boot = arrayfun(@(t) mean(T_boot(:,j,i) <= t),tau_grid);
        dist_max(j,i) = max(abs(G_boot - G_pop(tau_grid,n)));
        nexttile
        hold on
        plot(tau_grid,G_boot)
        plot(tau_grid,G_pop(tau_grid,n))
        title({['n = ' num2str(n) ',l = ' num2str(l)], ...
            ['Largest distance: ' num2str(dist_max(j,i),'%.3f')]})
        xlim([tau_grid(1) tau_grid(end)])
        if i == 1
            ylabel(block_label{j},'FontWeight','bold')
        end
        hold off
    end
end
x = xlabel(layout,'\tau');
y = ylabel(layout,'Cumulative probability');
x.FontSize = get(groot,'defaultAxesFontSize');
y.FontSize = get(groot,'defaultAxesFontSize');
lgd = legend( ...
    'Block bootstrap CDF', ...
    'Sampling CDF', ...
    'Orientation','horizontal');
lgd.Layout.Tile = 'south';
t = title(layout,['Fig. 2. Block bootstrap CDF and sampling CDF ' ...
    'as n grows (B = ' num2str(N_boot) ')']);
t.FontSize = get(groot,'defaultAxesFontSize');
t.FontWeight = 'normal';

%% 8. Block bootstrap variance across repeated samples

% Figures 5 and 6 are based on a single sample for each n. To see whether
% the block bootstrap converges in probability, compute the bootstrap
% variance of Tn* in many independent samples. Because all blocks have the
% same length, the bootstrap variance equals l times the mean squared
% deviation of the block means from the sample mean, so it can be computed
% exactly without drawing bootstrap samples.

% 8.1. Set the number of repeated samples
N_rep = 1000;

% 8.2. Preallocate storage for the ratio of the bootstrap variance to the
% true variance of Tn
var_ratio = NaN(N_rep,N_rules,N_n);

% 8.3. For each repeated sample and sample size, draw a new AR(1) series
% and compute the bootstrap variance for each block-length rule
for r = 1:N_rep
    for i = 1:N_n
        n = N_obs_vec(i);
        data_rep = ar1(random('Normal',0,sigma_eps,[n,1]));
        for j = 1:N_rules
            l = block_rule{j}(n);
            k = n / l;
            block_means = mean(reshape(data_rep,l,k),1)';
            var_boot = l * mean((block_means - mean(data_rep)).^2);
            var_ratio(r,j,i) = var_boot / v_pop(n);
        end
    end
end

% 8.4. Set the grid of ratio values at which the PDFs are evaluated
ratio_grid = linspace(0,3,3001)';

% 8.5. Plot the PDF of the variance ratio across repeated samples for each
% block-length rule and sample size. A ratio of 1 means that the bootstrap
% variance equals the true variance.
figure('Position',[100 100 1200 450]);
layout = tiledlayout(1,N_rules, ...
    'TileSpacing','compact','Padding','compact');
for j = 1:N_rules
    nexttile
    hold on
    for i = 1:N_n
        plot(ratio_grid,ksdensity(var_ratio(:,j,i),ratio_grid, ...
            'function','pdf','Support','positive', ...
            'BoundaryCorrection','reflection'))
    end
    xline(1);
    title(block_label{j})
    xlim([ratio_grid(1) ratio_grid(end)])
    hold off
end
x = xlabel(layout,'Bootstrap variance / true variance');
y = ylabel(layout,'Density');
x.FontSize = get(groot,'defaultAxesFontSize');
y.FontSize = get(groot,'defaultAxesFontSize');
lgd = legend( ...
    ['n = ' num2str(N_obs_vec(1))], ...
    ['n = ' num2str(N_obs_vec(2))], ...
    ['n = ' num2str(N_obs_vec(3))], ...
    'True variance','Orientation','horizontal');
lgd.Layout.Tile = 'south';
t = title(layout,['Fig. 3. Block bootstrap variance across ' ...
    num2str(N_rep) ' samples']);
t.FontSize = get(groot,'defaultAxesFontSize');
t.FontWeight = 'normal';

%% 9. Summarize the results

% Print, for each block-length rule and sample size, the block length, the
% number of blocks, the true variance of Tn, the bootstrap variance in the
% single sample of Figures 5 and 6, the largest distance between the CDFs,
% and the mean and standard deviation of the variance ratio across the
% repeated samples.
fprintf('\n%-12s %6s %6s %6s %8s %8s %8s %8s %8s\n','Rule', ...
    'n','l','k','v_n','Var*','Dist','MeanRat','SDRat')
for j = 1:N_rules
    for i = 1:N_n
        n = N_obs_vec(i);
        l = block_rule{j}(n);
        fprintf('%-12s %6d %6d %6d %8.3f %8.3f %8.3f %8.3f %8.3f\n', ...
            block_label{j},n,l,n/l,v_pop(n),var(T_boot(:,j,i)), ...
            dist_max(j,i),mean(var_ratio(:,j,i)),std(var_ratio(:,j,i)))
    end
end
fprintf('Variance of the AR(1) process (gamma_0): %.3f\n',gamma_0)
fprintf('Long-run variance sigma_eps^2/(1-phi)^2: %.3f\n', ...
    sigma_eps^2 / (1 - phi)^2)
