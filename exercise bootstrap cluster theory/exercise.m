% Exercise - Understanding the theory of cluster bootstrap using simulation

%% 1. Aim of the exercise

% This exercise illustrates when the cluster bootstrap approximates the
% sampling distribution of a statistic computed from clustered data. The
% data come from a one-way random-effects model, in which observations in
% the same cluster share a common cluster effect, and the statistic is the
% scaled sample mean. The exercise first shows that conventional inference,
% which treats the observations as independent, understates the variability
% of the sample mean. The cluster bootstrap resamples entire clusters, and
% its accuracy depends on the number of clusters G rather than on the
% number of observations n. Three cases are compared as n grows: the
% standard bootstrap, which resamples single observations and ignores the
% clustering; the cluster bootstrap when G grows with n; and the cluster
% bootstrap when G is fixed and only the clusters become larger. Finally,
% the exercise shows how unequal cluster sizes reduce the accuracy of the
% cluster bootstrap for a given number of clusters.

%% 2. Initialize MATLAB environment

% Clear workspace, command window, and open figures
clear;
clc;
close all;

%% 3. Construct the data-generating process and analytical distributions

% 3.1. Set a seed for the random number generator for reproducibility
rng(26)

% 3.2. Set the variance of the cluster effects
sigma2_alpha = 1;

% 3.3. Set the variance of the idiosyncratic errors
sigma2_eps = 4;

% 3.4. Compute the intra-cluster correlation
rho = sigma2_alpha / (sigma2_alpha + sigma2_eps);

% 3.5. Define a function that simulates clustered data for a vector of
% cluster sizes N_g. Each observation is the sum of the effect of its
% cluster and an idiosyncratic error, and the mean mu is 0.
draw_clusters = @(N_g) ...
    repelem( ...
    random('Normal',0,sqrt(sigma2_alpha),[length(N_g),1]),N_g(:) ...
    ) + ...
    random('Normal',0,sqrt(sigma2_eps),[sum(N_g),1]);

% 3.6. Define the variance of the statistic for a vector of cluster sizes
v_pop = @(N_g) length(N_g) * ...
    (sigma2_alpha * sum(N_g.^2) + sigma2_eps * sum(N_g)) / sum(N_g)^2;

% 3.7. Define the sampling PDF of the statistic
g_pop = @(tau,N_g) pdf('Normal',tau,0,sqrt(v_pop(N_g)));

% 3.8. Define the sampling CDF of the statistic
G_pop = @(tau,N_g) cdf('Normal',tau,0,sqrt(v_pop(N_g)));

% 3.9. Set the grid of tau values at which PDFs and CDFs are evaluated
tau_grid = linspace(-5,5,1000)';

% 3.10. Set the sample sizes
N_obs_vec = [200 1000 5000];

% The statistic of interest is TG = sqrt(G) * (sample mean - mu), with
% mu = 0. Because the cluster effects and the errors are normal, the
% sample mean is normally distributed, and the sampling distribution of TG
% is known analytically: TG ~ N(0, v_G). This allows the bootstrap
% approximation to be compared directly with the true sampling
% distribution.

%% 4. Conventional inference versus the sampling distribution

% Conventional inference estimates the variance of the sample mean by
% s^2/n, which treats the n observations as independent. For TG, this gives
% the conventional variance s^2/M, which converges to sigma^2/M, whereas
% the true variance is v_G = sigma2_alpha + sigma2_eps/M. Their ratio is
% the Moulton factor 1 + (M - 1)*rho. Compare the two distributions for
% several intra-cluster correlations, holding the total variance sigma^2
% and the cluster size M fixed. No random numbers are drawn here, so the
% results in the remaining sections do not depend on this section.

% 4.1. Set the total variance of an observation
sigma2 = sigma2_alpha + sigma2_eps;

% 4.2. Set the fixed cluster size used in conventional-inference comparison
M_conv = 10;

% 4.3. Set the intra-cluster correlations to compare
rho_vec = [0 0.1 0.2];

% 4.4. Compute the Moulton factor
moulton = 1 + (M_conv - 1) * rho_vec;

% 4.5. Compute the coverage of the nominal 95% confidence interval
coverage = 2 * cdf('Normal',1.96 ./ sqrt(moulton),0,1) - 1;

% 4.6. Plot the sampling PDF of TG and the normal PDF implied by
% conventional inference for each intra-cluster correlation. The title of
% each panel reports the Moulton factor and the coverage of a nominal 95%
% confidence interval based on the conventional standard error.
figure('Position',[100 100 1200 450]);
layout = tiledlayout(1,length(rho_vec), ...
    'TileSpacing','compact','Padding','compact');
for s = 1:length(rho_vec)
    v_true = rho_vec(s) * sigma2 + (1 - rho_vec(s)) * sigma2 / M_conv;
    v_conv = sigma2 / M_conv;
    nexttile
    hold on
    plot(tau_grid,pdf('Normal',tau_grid,0,sqrt(v_conv)))
    plot(tau_grid,pdf('Normal',tau_grid,0,sqrt(v_true)))
    title({['\rho = ' num2str(rho_vec(s))], ...
        ['Moulton factor: ' num2str(moulton(s),'%.1f')], ...
        ['Coverage of 95% CI: ' num2str(100 * coverage(s),'%.0f') '%']})
    xlim([tau_grid(1) tau_grid(end)])
    hold off
end
x = xlabel(layout,'\tau');
y = ylabel(layout,'Density');
x.FontSize = get(groot,'defaultAxesFontSize');
y.FontSize = get(groot,'defaultAxesFontSize');
lgd = legend( ...
    'Conventional normal approximation', ...
    'Sampling PDF', ...
    'Orientation','horizontal');
lgd.Layout.Tile = 'south';
t = title(layout,['Fig. 1. Sampling PDF and ' ...
    'conventional approximation (M = ' num2str(M_conv) ')']);
t.FontSize = get(groot,'defaultAxesFontSize');
t.FontWeight = 'normal';

%% 5. Define the asymptotic cases

% 5.1. Define the number of clusters as a function of the sample size
% for each case
G_rule = {@(n) n / 10, @(n) n / 10, @(n) 10};

% 5.2. Define the cluster size as a function of the sample size
% for each case
M_rule = {@(n) 10, @(n) 10, @(n) n / 10};

% In the first two cases, G grows with n and every cluster contains
% M = 10 observations. In the third case, G = 10 is fixed and cluster
% sizes increase as n grows.

% 5.3. Specify whether resampling is performed at the cluster level
resample_clusters = [false true true];

% 5.4. Define labels for the asymptotic cases
case_label = {'Standard','Cluster,M = 10','Cluster,G = 10'};

% 5.5. Set the number of asymptotic cases
N_cases = length(G_rule);

% 5.6. Set the number of sample sizes considered
N_n = length(N_obs_vec);

% Store simulated data in matrix form. Columns represent clusters and rows
% represent observations within clusters. For the first two cases, samples
% are obtained by selecting the first G columns of data_G. For the third
% case, samples are obtained by selecting the first M rows of data_M.

% 5.7. Generate clustered data for the cases in which G grows
% with the sample size
G_max = max(N_obs_vec) / 10;
data_G = reshape(draw_clusters(repmat(10,G_max,1)),10,G_max);

% 5.8. Generate clustered data for the case in which cluster size grows
% with the sample size
M_max = max(N_obs_vec) / 10;
data_M = reshape(draw_clusters(repmat(M_max,10,1)),M_max,10);

%% 6. Construct bootstrap distributions for the asymptotic cases

% 6.1. Set the number of bootstrap replications
N_boot = 10^4;

% 6.2. Preallocate storage for the bootstrap statistic
T_boot = NaN(N_boot,N_cases,N_n);

% 6.3. For each asymptotic case and sample size, select the original
% sample, generate bootstrap samples by resampling observations or
% clusters, and compute the bootstrap statistic
for j = 1:N_cases
    for i = 1:N_n
        n = N_obs_vec(i);
        G = G_rule{j}(n);
        M = M_rule{j}(n);
        if j < 3
            data_sample = data_G(:,1:G);
        else
            data_sample = data_M(1:M,:);
        end
        for b = 1:N_boot
            if resample_clusters(j)
                % Draw G clusters (columns) with replacement
                data_boot = datasample(data_sample,G,2,'Replace',true);
            else
                % Draw n observations with replacement
                data_boot = datasample(data_sample(:),n,'Replace',true);
            end
            % TG* = sqrt(G) * (bootstrap sample mean - original sample mean)
            T_boot(b,j,i) = sqrt(G)*(mean(data_boot(:)) - ...
                mean(data_sample(:)));
        end
    end
end

%% 7. Convergence of the bootstrap PDF to the sampling PDF

% Plot the bootstrap PDF and the sampling PDF for each asymptotic case
% (rows) and sample size (columns).
figure('Position',[100 100 1000 1000]);
layout = tiledlayout(N_cases,N_n, ...
    'TileSpacing','compact','Padding','compact');
for j = 1:N_cases
    for i = 1:N_n
        n = N_obs_vec(i);
        G = G_rule{j}(n);
        M = M_rule{j}(n);
        nexttile
        hold on
        plot(tau_grid,ksdensity(T_boot(:,j,i),tau_grid,'function','pdf'))
        plot(tau_grid,g_pop(tau_grid,repmat(M,G,1)))
        title(['G = ' num2str(G) ',M = ' num2str(M)])
        xlim([tau_grid(1) tau_grid(end)])
        if i == 1
            ylabel(case_label{j},'FontWeight','bold')
        end
        hold off
    end
end
x = xlabel(layout,'\tau');
y = ylabel(layout,'Density');
x.FontSize = get(groot,'defaultAxesFontSize');
y.FontSize = get(groot,'defaultAxesFontSize');
lgd = legend( ...
    'Bootstrap PDF', ...
    'Sampling PDF', ...
    'Orientation','horizontal');
lgd.Layout.Tile = 'south';
t = title(layout,['Fig. 2. Bootstrap PDF and sampling PDF ' ...
    'as n grows (B = ' num2str(N_boot) ')']);
t.FontSize = get(groot,'defaultAxesFontSize');
t.FontWeight = 'normal';

%% 8. Convergence of the bootstrap CDF to the sampling CDF

% Plot the bootstrap CDF and the true sampling CDF for each case (rows)
% and sample size (columns). The title of each panel reports the largest
% distance between the two CDFs over the grid of tau values.
dist_max = NaN(N_cases,N_n);
figure('Position',[100 100 1000 1000]);
layout = tiledlayout(N_cases,N_n, ...
    'TileSpacing','compact','Padding','compact');
for j = 1:N_cases
    for i = 1:N_n
        n = N_obs_vec(i);
        G = G_rule{j}(n);
        M = M_rule{j}(n);
        G_boot = arrayfun(@(t) mean(T_boot(:,j,i) <= t),tau_grid);
        dist_max(j,i) = max(abs(G_boot - G_pop(tau_grid,repmat(M,G,1))));
        nexttile
        hold on
        plot(tau_grid,G_boot)
        plot(tau_grid,G_pop(tau_grid,repmat(M,G,1)))
        title({['G = ' num2str(G) ',M = ' num2str(M)], ...
            ['Largest distance: ' num2str(dist_max(j,i),'%.3f')]})
        xlim([tau_grid(1) tau_grid(end)])
        if i == 1
            ylabel(case_label{j},'FontWeight','bold')
        end
        hold off
    end
end
x = xlabel(layout,'\tau');
y = ylabel(layout,'Cumulative probability');
x.FontSize = get(groot,'defaultAxesFontSize');
y.FontSize = get(groot,'defaultAxesFontSize');
lgd = legend( ...
    'Bootstrap CDF', ...
    'Sampling CDF', ...
    'Orientation','horizontal');
lgd.Layout.Tile = 'south';
t = title(layout,['Fig. 3. Bootstrap CDF and sampling CDF ' ...
    'as n grows (B = ' num2str(N_boot) ')']);
t.FontSize = get(groot,'defaultAxesFontSize');
t.FontWeight = 'normal';

%% 9. Bootstrap variance across repeated samples

% Compute the bootstrap variance of TG* in many independent samples. With
% equal cluster sizes, it can be computed exactly without drawing bootstrap
% samples. For the standard bootstrap, it equals the variance of the
% observations divided by M. For the cluster bootstrap, it equals the mean
% squared deviation of the cluster means from the sample mean.

% 9.1. Set the number of repeated samples
N_rep = 1000;

% 9.2. Preallocate storage for the ratio of the bootstrap variance to the
% true variance of TG
var_ratio = NaN(N_rep,N_cases,N_n);

% 9.3. For each repeated sample, case and sample size, draw a new sample
% and compute the bootstrap variance
for r = 1:N_rep
    for j = 1:N_cases
        for i = 1:N_n
            n = N_obs_vec(i);
            G = G_rule{j}(n);
            M = M_rule{j}(n);
            data_rep = reshape(draw_clusters(repmat(M,G,1)),M,G);
            if resample_clusters(j)
                var_boot = mean((mean(data_rep,1) - mean(data_rep(:))).^2);
            else
                var_boot = mean((data_rep(:) - mean(data_rep(:))).^2) / M;
            end
            var_ratio(r,j,i) = var_boot / v_pop(repmat(M,G,1));
        end
    end
end

% 9.4. Set the grid of ratio values at which the PDFs are evaluated
ratio_grid = linspace(0,3,3001)';

% 9.5. Plot the PDF of the variance ratio across repeated samples for each
% case and sample size. A ratio of 1 means that the bootstrap variance
% equals the true variance. The panels are stacked vertically, one per
% case, so that each panel is wide enough to show the narrow distributions.
figure('Position',[100 100 1000 1000]);
layout = tiledlayout(N_cases,1, ...
    'TileSpacing','compact','Padding','compact');
for j = 1:N_cases
    nexttile
    hold on
    for i = 1:N_n
        plot(ratio_grid,ksdensity(var_ratio(:,j,i),ratio_grid, ...
            'function','pdf','Support','positive', ...
            'BoundaryCorrection','reflection'))
    end
    xline(1);
    title(case_label{j})
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
t = title(layout,['Fig. 4. Bootstrap variance across ' ...
    num2str(N_rep) ' samples']);
t.FontSize = get(groot,'defaultAxesFontSize');
t.FontWeight = 'normal';

%% 10. Unequal cluster sizes

% Keep the number of clusters and the number of observations fixed, and let
% the cluster sizes become increasingly unequal. The size of cluster g is
% proportional to g^(-gamma), so gamma = 0 gives equal sizes and a larger
% gamma concentrates the observations in a few large clusters.

% 10.1. Set the number of clusters
G_het = 50;

% 10.2. Set the sample size
n_het = 2500;

% 10.3. Set the values of gamma
gamma_vec = [0 0.5 1];

% 10.4. Set the number of clusters, the sample size and the values of gamma
N_het = length(gamma_vec);

% 10.5. Compute the cluster sizes for each value of gamma, rounding so that
% the sizes are integers that sum to n_het
N_g_het = cell(1,N_het);
for s = 1:N_het
    w = (1:G_het)'.^(-gamma_vec(s));
    N_g = max(1,floor(w / sum(w) * n_het));
    while sum(N_g) < n_het
        [~,g] = max(w / sum(w) * n_het - N_g);
        N_g(g) = N_g(g) + 1;
    end
    N_g_het{s} = N_g;
end

% 10.6. Set the number of bootstrap replications per sample
N_boot_het = 1000;

% 10.7. For each repeated sample and size distribution, draw a new sample,
% draw bootstrap samples of clusters, and compute the ratio of the
% bootstrap variance to the true variance. With unequal cluster sizes, the
% number of observations varies across bootstrap samples, so the bootstrap
% variance is simulated rather than computed exactly.
var_ratio_het = NaN(N_rep,N_het);
for r = 1:N_rep
    for s = 1:N_het
        N_g = N_g_het{s};
        data_rep = draw_clusters(N_g);
        cluster_id = repelem((1:G_het)',N_g);
        totals = accumarray(cluster_id,data_rep);
        idx = randi(G_het,G_het,N_boot_het);
        mean_boot = sum(totals(idx),1) ./ sum(N_g(idx),1);
        var_ratio_het(r,s) = G_het * var(mean_boot) / v_pop(N_g);
    end
end

% 10.8. Plot the PDF of the variance ratio for each size distribution
het_label = cell(1,N_het);
figure('Position',[100 100 1000 1000]);
hold on
for s = 1:N_het
    plot(ratio_grid,ksdensity(var_ratio_het(:,s),ratio_grid, ...
        'function','pdf','Support','positive', ...
        'BoundaryCorrection','reflection'))
    het_label{s} = ['\gamma = ' num2str(gamma_vec(s)) ',' ...
        'largest cluster ' num2str(100 * max(N_g_het{s}) / ...
        n_het,'%.0f') '% of n'];
end
xline(1,'k--');
xlim([ratio_grid(1) ratio_grid(end)])
xlabel('Bootstrap variance / true variance')
ylabel('Density')
legend([het_label, {'True variance'}],'Location','northeast')
title(['Fig. 5. Unequal cluster sizes (G = ' num2str(G_het) ...
    ', n = ' num2str(n_het) ')'])
hold off

%% 11. Summarize the results

% Print, for each case and sample size, the number of clusters, the cluster
% size, the true variance of TG, the bootstrap variance in the single
% sample of Figures 8 and 9, the largest distance between the CDFs, and the
% mean and standard deviation of the variance ratio across the repeated
% samples. Then print the same summary for unequal cluster sizes.
fprintf('\n%-16s %6s %5s %5s %8s %8s %8s %8s %8s\n', ...
    'Case','n','G','M','v_G','Var*','Dist','MeanRat','SDRat')
for j = 1:N_cases
    for i = 1:N_n
        n = N_obs_vec(i);
        G = G_rule{j}(n);
        M = M_rule{j}(n);
        fprintf('%-16s %6d %5d %5d %8.3f %8.3f %8.3f %8.3f %8.3f\n', ...
            case_label{j}, ...
            n, ...
            G, ...
            M, ...
            v_pop(repmat(M,G,1)), ...
            var(T_boot(:,j,i)), ...
            dist_max(j,i), ...
            mean(var_ratio(:,j,i)), ...
            std(var_ratio(:,j,i)))
    end
end
fprintf('\n%-8s %8s %8s %8s %8s\n','gamma','Largest','Share', ...
    'MeanRat','SDRat')
for s = 1:N_het
    fprintf('%-8.1f %8d %8.3f %8.3f %8.3f\n',gamma_vec(s), ...
        max(N_g_het{s}),max(N_g_het{s}) / n_het, ...
        mean(var_ratio_het(:,s)),std(var_ratio_het(:,s)))
end
fprintf('\n%-8s %10s %10s\n','rho','Moulton','Coverage')
for s = 1:length(rho_vec)
    fprintf('%-8.1f %10.2f %10.3f\n',rho_vec(s),moulton(s),coverage(s))
end
fprintf('Intra-cluster correlation rho: %.3f\n',rho)
fprintf('Moulton factor for M = 10: %.3f\n',1 + 9 * rho)
