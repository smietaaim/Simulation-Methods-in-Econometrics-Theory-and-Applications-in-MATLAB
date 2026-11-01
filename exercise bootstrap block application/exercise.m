% Exercise - Understanding the method of block bootstrap

%% 1. Aim of the exericse

% The aim of this exercise is to understand how to correctly estimate the
% sampling distribution of a statistic (e.g., the sample mean) when data
% are serially correlated, such as in time-series settings. Standard
% bootstrap methods assume independent observations and can lead to biased
% inference. We use the block bootstrap method to account for dependence
% across observations and compare its performance to sampling directly from
% the population.

%% 2. Theory

% Refer to the accompanying PDF file for the theory.

%% 3. Set values for the parameters of the simulation

% 3.1. Clear the memory
clear;

% 3.2. Set the number of simulations as the number of bootstrap samples
N_sim = 5000;

%% 4. Generate population data

% 4.1. Set the population size
N_obs_pop = 100000;

% 4.2. Set AR(1) persistence parameter
phi = 0.7;

% 4.3. Generate white noise innovations
varepsilon = random('Normal',0,1,[N_obs_pop,1]);

% 4.4. Simulate AR(1) time‑series data
data_pop = filter(1,1-phi,varepsilon);

%% 5. Plot AR(1) time series

% Figure
figure('Position',[100 100 1000 1000]);
plot(1:100,data_pop(1:100));
xlabel('Time (t)');
ylabel('X_t');
title('Fig. 1. First 100 points of the AR(1) series');
legend('AR(1) process');

%% 6. Draw samples from the population

% 6.1. Set the sample size
N_obs_sample = 5000;

% 6.2. Preallocate matrix to store samples
data_samples_pop = NaN(N_obs_sample,N_sim);

% 6.3. Preallocate vector to store sample means
means_data_samples_pop = NaN(N_sim,1);

% 6.4. Sample from the population and compute the sample mean each time
for i = 1:N_sim
    start_idx = randi(N_obs_pop-N_obs_sample+1);
    data_samples_pop(:,i) = data_pop(start_idx:start_idx+N_obs_sample-1);
    means_data_samples_pop(i) = mean(data_samples_pop(:,i));
end

%% 7. Pick an "initial" sample

% 7.1. Randomly pick one sample index
sample_index = randi(N_sim);

% 7.2. Pick one sample from the previously drawn samples
data_sample = data_samples_pop(:,sample_index);

% 7.3. Compute optimal block length
block_length = ceil(N_obs_sample^(1/5));

% 7.4. Compute how many full blocks can be formed
N_blocks = floor(N_obs_sample/block_length); % Max. num. of full blocks

% 7.5. Trim sample to fit an exact number of full blocks
trimmed_sample = data_sample(1:N_blocks*block_length);

% 7.6. Reshape trimmed sample into non-overlapping blocks
blocks = reshape(trimmed_sample,block_length,N_blocks)';

%% 8. Draw (bootstrap) samples from the initial sample 

% 8.1. Preallocate vector to store (bootstrap) sample means
means_data_samples_boot = NaN(N_sim,1);

% 8.2. Resample k blocks from the initial sample and compute the mean
for i = 1:N_sim
    data_samples_boot = datasample(blocks,N_blocks,1,'Replace',true);
    data_samples_boot_flat = data_samples_boot(:);
    means_data_samples_boot(i) = mean(data_samples_boot_flat);
end

%% 9. Plot estimated PDFs

% 9.1. Estimate the density
[f_boot,x_boot] = ksdensity(means_data_samples_boot,'function','pdf');

% 9.2. Create a figure
figure('Position',[100 100 1000 1000]);
hold on
plot(x_boot,f_boot,'Color',[0.8500,0.3250,0.0980], ...
    'DisplayName','Bootstrap sample means');
[f_pop,x_pop] = ksdensity(means_data_samples_pop,'function','pdf');
plot(x_pop,f_pop,'Color',[0,0.4470,0.7410], ...
    'DisplayName','Population sample means');
title(['Fig. 2. PDFs of sample means based on bootstrap and ' ...
    'population sampling']);
ylabel('Density');
xlabel('Sample mean');
legend('show')
hold off

%% 10. Plot estimated CDFs

% 10.1. Estimate the density
[f_boot,x_boot] = ksdensity(means_data_samples_boot,'function','cdf');

% 10.2. Create a figure
figure('Position',[100 100 1000 1000]);
hold on
plot(x_boot,f_boot,'Color',[0.8500,0.3250,0.0980], ...
    'DisplayName','Bootstrap sample means');
[f_pop,x_pop] = ksdensity(means_data_samples_pop,'function','cdf');
plot(x_pop,f_pop,'Color',[0,0.4470,0.7410], ...
    'DisplayName','Population sample means');
title(['Fig. 3. CDFs of sample means based on bootstrap and ' ...
    'population sampling']);
ylabel('Cumulative distribution');
xlabel('Sample mean');
legend('show');
hold off
