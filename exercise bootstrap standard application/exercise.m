% Exercise - Understanding the method of basic bootsrap

%% 1. Aim of the exercise
% The aim of this exercise is to understand how to correctly estimate the
% sampling distribution of a statistic, in particular that of the sample
% mean, when the population distribution is unknown. We compare two
% approaches: (i) bootstrap resampling from the sample, and (ii) repeated
% sampling from the full population, to evaluate how well the bootstrap
% approximates the true distribution.

%% 2. Theory
% Refer to the accompanying PDF file for the theory.

%% 3. Set values for the parameters of the simulation

% 3.1. Clear the memory 
clear;

% 3.2. Set the number of simulations as the number of samples
N_sim = 1000;

%% 4. Generate population data

% 4.1. Set the population size
N_obs_pop = 1000;

% 4.2. Generate population data
data_pop = random('Normal',4,5,[N_obs_pop,1]);

%% 5. Draw samples from the population

% 5.1. Set the sample size
N_obs_sample = 100;

% 5.2. Preallocate matrix to store samples
data_samples_pop = NaN(N_obs_sample,N_sim);

% 5.3. Preallocate vector to store sample means
means_data_samples_pop = NaN(N_sim,1);

% 5.4. Draw samples from the population and compute their means
for i = 1:N_sim
    data_samples_pop(:,i) = datasample(data_pop,N_obs_sample, ...
        'Replace',false);
    means_data_samples_pop(i) = mean(data_samples_pop(:,i));
end

%% 6. Pick an "initial" sample

% 6.1. Randomly pick one sample index
sample_index = randi(N_sim);  

% 6.2. Pick one sample from the previously drawn samples
data_sample = data_samples_pop(:,sample_index);

%% 7. Draw (bootsrap) samples from the initial sample

% 7.1. Preallocate vector to store (bootsrap) sample means
means_data_samples_boot = NaN(N_sim,1);

% 7.2. Draw samples from the initial sample and compute their means
for i = 1:N_sim
    data_samples_boot = datasample(data_sample,N_obs_sample, ...
        'Replace',true);
    means_data_samples_boot(i) = mean(data_samples_boot);
end

%% 8. Plot the PDFs of sample means from bootstrap and population sampling
figure;
hold on
ksdensity(means_data_samples_boot,'function','pdf');
ksdensity(means_data_samples_pop,'function','pdf');
ylabel('Density');
xlabel('Sample mean');
legend('Population sampling','Bootstrap sampling');
title(['Fig. 1. PDFs of sample means based on bootstrap and ' ...
    'population sampling']);
hold off

%% 9. Plot the CDFs of sample means from bootstrap and population sampling
figure;
hold on
ksdensity(means_data_samples_boot,'function','cdf');
ksdensity(means_data_samples_pop,'function','cdf');
ylabel('Cumulative distribution');
xlabel('Sample mean');
legend('Population sampling','Bootstrap sampling');
title(['Fig. 2. CDFs of sample means based on bootsrap and ' ...
    'population sampling']);
hold off
