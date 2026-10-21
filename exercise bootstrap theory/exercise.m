% Exercise - Understanding bootstrap theory using simulation

%% 1. Aim of the exercise

% This exercise illustrates two key convergence results that explain why
% the bootstrap works. First, as the number of bootstrap replications B
% increases, the simulated bootstrap distribution converges to the
% underlying bootstrap distribution. Second, as the sample size n
% increases, the bootstrap distribution converges to the true sampling
% distribution. The first result ensures that the bootstrap distribution is
% accurately simulated, while the second ensures that it provides a valid
% approximation to the unknown sampling distribution of the statistic.

%% 2. Initialize MATLAB environment

% Clear workspace, command window, and open figures
clear;
clc;
close all;

%% 3. Construct the data-generating process and analytical distributions

% 3.1. Set a seed for the random number generator for reproducibility
rng(26)

% 3.2. Set the mean of the exponential distribution generating the data
mu = 2;

% 3.3. Define the sampling PDF of the statistic
g_pop = @(tau,n) pdf('Gamma',mu + tau / sqrt(n),n,mu / n)/sqrt(n);

% 3.4. Define the sampling CDF of the statistic
G_pop = @(tau,n) cdf('Gamma',mu + tau / sqrt(n),n,mu / n);

% 3.5. Set the grid of tau values at which PDFs and CDFs are evaluated
tau_grid = linspace(-6,8,1000)';

% 3.6. Set the sample sizes
N_obs_vec = [10 100 1000];

% 3.7. Draw a sequence of observations from the population
data_all = random('Exponential',mu,[max(N_obs_vec),1]);

% The statistic of interest is Tn = sqrt(n) * (sample mean - mu). Because
% the sample mean of n exponential draws follows a Gamma distribution, the
% sampling distribution of Tn is known analytically. This allows the
% bootstrap approximation to be compared directly with the true sampling
% distribution.

%% 4. Construct bootstrap distributions for different numbers of replications

% Prepare to study how the simulated bootstrap distribution converges to
% the bootstrap distribution as the number of bootstrap replications B
% increases.

% 4.1. Set the sample size
N_obs = 20;

% 4.2. Select the original sample
data_sample = data_all(1:N_obs);

% 4.3. Set the numbers of bootstrap replications to compare
N_boot_vec = [10 100 10000];

% 4.4. Set the benchmark number of bootstrap replications
N_boot_ref = 10^5;

% A large number of bootstrap replications is used to approximate the
% underlying bootstrap distribution.

% 4.5. Preallocate storage for the bootstrap statistic
T_boot = NaN(N_boot_ref,1);

% 4.6. Draw bootstrap samples and compute the bootstrap statistic
for b = 1:N_boot_ref
    data_boot = datasample(data_sample,N_obs,'Replace',true);
    % Tn* = sqrt(n) * (bootstrap sample mean - original sample mean)
    T_boot(b) = sqrt(N_obs)*(mean(data_boot) - mean(data_sample));
end

% 4.7. Approximate the bootstrap PDF
g_boot_ref = ksdensity(T_boot,tau_grid,'function','pdf');

% 4.8. Approximate the bootstrap CDF
G_boot_ref = arrayfun(@(t) mean(T_boot <= t),tau_grid);

% The bootstrap CDF at tau equals the proportion of bootstrap statistics
% that do not exceed tau.

%% 5. Convergence from simulated bootstrap to bootstrap distribution

% Plot the simulated bootstrap PDF based on the first B bootstrap
% replications, together with the benchmark bootstrap PDF and the true
% sampling PDF.
figure('Position',[100 100 1000 1000]);
layout = tiledlayout(1,3,'TileSpacing','compact','Padding','compact');
for i = 1:length(N_boot_vec)
    B = N_boot_vec(i);
    nexttile
    hold on
    plot(tau_grid,ksdensity(T_boot(1:B),tau_grid,'function','pdf'))
    plot(tau_grid,g_boot_ref,'r')
    plot(tau_grid,g_pop(tau_grid,N_obs),'k--')
    title(['B = ' num2str(B)])
    xlim([tau_grid(1) tau_grid(end)])
    hold off
end
xlabel(layout,'\tau')
ylabel(layout,'Density')
lgd = legend( ...
    'Simulated bootstrap PDF (first B samples)', ...
    'Bootstrap PDF (10^5 samples)', ...
    'Sampling PDF', ...
    'Orientation','horizontal');
lgd.Layout.Tile = 'south';
title(layout,['Fig. 1. Simulated bootstrap PDF as B grows (n = ' ...
    num2str(N_obs) ')'])

% Plot the simulated bootstrap CDF based on the first B bootstrap
% replications, together with the benchmark bootstrap CDF and the true
% sampling CDF. The title of each panel reports the largest distance
% between the simulated bootstrap CDF and the benchmark bootstrap CDF over
% the grid of tau values.
figure('Position',[100 100 1000 1000]);
layout = tiledlayout(1,3,'TileSpacing','compact','Padding','compact');
for i = 1:length(N_boot_vec)
    B = N_boot_vec(i);
    G_boot = arrayfun(@(t) mean(T_boot(1:B) <= t),tau_grid);
    dist_max = max(abs(G_boot - G_boot_ref));
    nexttile
    hold on
    plot(tau_grid,G_boot)
    plot(tau_grid,G_boot_ref,'r')
    plot(tau_grid,G_pop(tau_grid,N_obs),'k--')
    title({['B = ' num2str(B)], ...
        ['Largest distance: ' num2str(dist_max,'%.3f')]})
    xlim([tau_grid(1) tau_grid(end)])
    hold off
end
xlabel(layout,'\tau')
ylabel(layout,'Cumulative probability')
lgd = legend( ...
    'Simulated bootstrap CDF (first B samples)', ...
    'Bootstrap CDF (10^5 samples)', ...
    'Sampling CDF', ...
    'Orientation','horizontal');
lgd.Layout.Tile = 'south';
title(layout,['Fig. 2. Simulated bootstrap CDF as B grows (n = ' ...
    num2str(N_obs) ')'])

%% 6. Construct bootstrap distributions for different sample sizes

% Prepare to study how the bootstrap distribution converges to the sampling
% distribution as the sample size n increases.

% 6.1. Set the number of bootstrap replications
N_boot = 10^4;

% 6.2. Preallocate storage for the bootstrap statistic
T_boot_n = NaN(N_boot,length(N_obs_vec));

% 6.3. For each sample size, select the original sample, draw bootstrap
% replications, and compute the bootstrap statistic
for i = 1:length(N_obs_vec)
    n = N_obs_vec(i);
    data_sample = data_all(1:n);
    for b = 1:N_boot
        data_boot = datasample(data_sample,n,'Replace',true);
        T_boot_n(b,i) = sqrt(n)*(mean(data_boot) - mean(data_sample));
    end
end

%% 7. Convergence from bootstrap distribution to sampling distribution

% Plot the bootstrap PDF and the true sampling PDF for different sample
% sizes. As n increases, the bootstrap PDF should provide an increasingly
% accurate approximation to the sampling PDF.
figure('Position',[100 100 1000 1000]);
layout = tiledlayout(1,3,'TileSpacing','compact','Padding','compact');
for i = 1:length(N_obs_vec)
    n = N_obs_vec(i);
    nexttile
    hold on
    plot(tau_grid,ksdensity(T_boot_n(:,i),tau_grid,'function','pdf'),'r')
    plot(tau_grid,g_pop(tau_grid,n),'k--')
    title(['n = ' num2str(n)])
    xlim([tau_grid(1) tau_grid(end)])
    hold off
end
xlabel(layout,'\tau')
ylabel(layout,'Density')
lgd = legend( ...
    'Bootstrap PDF', ...
    'Sampling PDF', ...
    'Orientation', ...
    'horizontal');
lgd.Layout.Tile = 'south';
title(layout,['Fig. 3. Bootstrap PDF and sampling PDF as n grows (B = ' ...
    num2str(N_boot) ')'])

% Plot the bootstrap CDF and the true sampling CDF for different sample
% sizes. The title of each panel reports the largest distance between the
% two CDFs over the grid of tau values. As n increases, the bootstrap CDF
% should provide an increasingly accurate approximation to the sampling
% CDF.
figure('Position',[100 100 1000 1000]);
layout = tiledlayout(1,3,'TileSpacing','compact','Padding','compact');
for i = 1:length(N_obs_vec)
    n = N_obs_vec(i);
    G_boot = arrayfun(@(t) mean(T_boot_n(:,i) <= t),tau_grid);
    dist_max = max(abs(G_boot - G_pop(tau_grid,n)));
    nexttile
    hold on
    plot(tau_grid,G_boot,'r')
    plot(tau_grid,G_pop(tau_grid,n),'k--')
    title({['n = ' num2str(n)], ...
        ['Largest distance: ' num2str(dist_max,'%.3f')]})
    xlim([tau_grid(1) tau_grid(end)])
    hold off
end
xlabel(layout,'\tau')
ylabel(layout,'Cumulative probability')
lgd = legend( ...
    'Bootstrap CDF', ...
    'Sampling CDF', ...
    'Orientation', ...
    'horizontal');
lgd.Layout.Tile = 'south';
title(layout,['Fig. 4. Bootstrap CDF and sampling CDF as n grows (B = ' ...
    num2str(N_boot) ')'])
