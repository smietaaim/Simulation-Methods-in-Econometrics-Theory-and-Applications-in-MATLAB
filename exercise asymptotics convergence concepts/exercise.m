% Exercise - Understanding convergence concepts using simulation

%% 1. Aim of the exercise

% This exercise illustrates convergence concepts that underlie many
% econometric derivations.

%% 2. Initialize MATLAB environment

% Clear workspace, command window, and open figures
clear;
clc;
close all;

%% 3. Set the parameters of the simulation

% 3.1. Define alternative sample sizes (number of coin tosses)
N_tosses = 1:9:1000;

% 3.2. Define the number of alternative sample sizes
N_samples = length(N_tosses);

% 3.3. Define the number of simulated samples
N_sim = 1000;

%% 4. Define the population parameters

% 4.1. Define the mean for a fair coin
mean_population = 0.5;

% 4.2. Define the variance for a fair coin
var_population = 0.25;

%% 5. Generate data and calculate the sample mean

% 5.1. Preallocate memory for the simulated sample means
means_samples = NaN(N_sim,N_samples);

% 5.2. Simulate coin tosses and calculate the sample mean
for j = 1:N_samples
    % Generate N_sim independent samples of size N_tosses(j)
    sequence_of_tosses = random('Binomial',1,0.5,[N_tosses(j),N_sim]);
    % Calculate the sample mean for each simulated sample
    means_samples(:,j) = mean(sequence_of_tosses,1)';
end

%% 6. Visualizing the convergence rate

% 6.1. Calculate the average absolute distance at each sample size
mean_abs_error = mean(abs(means_samples - mean_population),1);

% 6.2. Compute the scale factor for the candidate 1/sqrt(n) decay curve
root_n_curve_scale = sqrt(N_tosses(1)) * mean_abs_error(1);

% 6.3. Generate the candidate 1/sqrt(n) decay curve
root_n_decay_curve = root_n_curve_scale ./ sqrt(N_tosses);

% 6.4. Compute the scale factor for the candidate 1/n decay curve
one_over_n_curve_scale = N_tosses(1) * mean_abs_error(1);

% 6.5. Generate the candidate 1/n decay curve
one_over_n_decay_curve = one_over_n_curve_scale ./ N_tosses;

% 6.6. Plot the empirical error path and candidate overlays
figure('Position',[1000 1000 1000 1000]);
hold on
plot(N_tosses,mean_abs_error,'.', ...
    'Color',[0.000 0.000 0.000], ...
    'DisplayName','Simulated average absolute error');
plot(N_tosses,root_n_decay_curve,'-', ...
    'Color',[0.000 0.447 0.741], ...
    'DisplayName','Candidate decay: 1/\surd{n}');
plot(N_tosses,one_over_n_decay_curve,'-', ...
    'Color',[0.301 0.745 0.933], ...
    'DisplayName','Candidate decay: 1/n');
title('Fig 1. Convergence rate of the sample proportion');
ylabel(['Mean absolute deviation of the sample proportion from ' ...
    'the true probability'])
xlabel('Sample size');
legend('show');
hold off

%% 7. Select representative sample sizes for visualization

% 7.1. Define representative fractions of the sample-size grid
fractions = [0.10, 0.30, 0.50, 1.00];

% 7.2. Locate the corresponding positions in the sample-size grid
target_indices = max(1,round(fractions * N_samples));

% 7.3. Extract the corresponding sample sizes
target_n = N_tosses(target_indices);

%% 8. Visualizing convergence in probability: op(1)

% Plot the unscaled estimation errors
figure('Position',[1000 1000 1000 1000]);
hold on
for k = 1:length(target_indices)
    % The column index in means_samples
    j = target_indices(k);
    % The exact sample size for this iteration
    plot_n = N_tosses(j);
    % Compute the unscaled estimation error
    unscaled_error_col = means_samples(:,j) - mean_population;
    % Replicate the sample size value for all Monte Carlo simulations
    x_axis_values = repmat(plot_n,N_sim,1);
    % Plot the unscaled estimation errors for this sample size
    scatter(x_axis_values,unscaled_error_col,5,'.', ...
        'MarkerEdgeColor',[0.000 0.000 0.000], ...
        'MarkerEdgeAlpha',0.25, ...
        'HandleVisibility','off');
end
% Construct the theoretical 95% decay envelope
n_continuous = linspace(min(N_tosses),max(N_tosses)*1.05,500);
theoretical_bound = ...
    (1.96 * sqrt(var_population)) ./ sqrt(n_continuous);
% Overlay the theoretical decay envelope
plot(n_continuous,theoretical_bound,'-', ...
    'Color',[0.850 0.325 0.098]);
plot(n_continuous,-theoretical_bound,'-', ...
    'Color',[0.850 0.325 0.098], ...
    'HandleVisibility','off');
% Add a reference line at zero estimation error
yline(0,'-', ...
    'Color',[0.000 0.000 0.000], ...
    'HandleVisibility','off');
title('Fig 2. Convergence in probability of the unscaled error');
ylabel(['Unscaled deviation of the sample proportion from the true ' ...
    'probability']);
xlabel('Sample size');
% Dynamically set the x-axis limit based on the maximum sample size
xlim([0,max(N_tosses)*1.05]);
% Display only the selected sample sizes on the x-axis
xticks(target_n);
hold off

%% 9. Visualizing stochastic boundedness: Op(1)

% Plot the root-n scaled estimation errors
figure('Position',[1000 1000 1000 1000]);
hold on
for k = 1:length(target_indices)
    % The column index in means_samples
    j = target_indices(k);
    % The exact sample size for this iteration
    plot_n = N_tosses(j);
    % Compute the root-n scaled estimation error
    scaled_error_col = sqrt(plot_n) * ...
        (means_samples(:,j) - mean_population);
    % Replicate the sample size value for all Monte Carlo simulations
    x_axis_values = repmat(plot_n,N_sim,1);
    % Plot the root-n scaled estimation errors for this sample size
    scatter(x_axis_values,scaled_error_col,5,'.', ...
    'MarkerEdgeColor',[0.000 0.000 0.000], ...
    'MarkerEdgeAlpha',0.25);
end
% Reference line corresponding to zero estimation error
yline(0,'-', ...
    'Color',[0.850 0.325 0.098]);
title('Fig 3. Stochastic boundedness of the root-n scaled error');
ylabel(['Root-n scaled deviation of the sample proportion from ' ...
    'the true probability'])
xlabel('Sample size');
% Dynamically set the x-axis limit based on the actual max value
xlim([0,max(N_tosses) * 1.05]);
% Fix the y-axis range
ylim([-2,2]);
% Display only the selected sample sizes on the x-axis
xticks(target_n);
hold off

%% 10. Visualizing asymptotic normality of the root-n scaled error

% 10.1. Select the largest sample size for the CLT demonstration
n_large = N_tosses(N_samples);

% 10.2. Calculate the root-n scaled estimation error
scaled_error_clt = sqrt(n_large) * ...
    (means_samples(:,N_samples) - mean_population);

% 10.3. Estimate the empirical density of the scaled errors
[f_empirical,x_empirical] = ksdensity(scaled_error_clt);

% 10.4. Calculate the theoretical standard deviation implied by the CLT
std_dev = sqrt(var_population);

% 10.5. Create an evaluation grid for the theoretical Normal density
x_theoretical = linspace(min(x_empirical),max(x_empirical),500);

% 10.6. Evaluate the theoretical Normal density on the grid
f_theoretical = normpdf(x_theoretical,0,std_dev);

% 10.7. Compare the empirical and theoretical densities
figure('Position',[1000 1000 1000 1000]);
hold on
% Plot the empirical kernel density estimate
plot(x_empirical,f_empirical,'-', ...
    'Color',[0.000 0.000 0.000], ...
    'DisplayName','Empirical density of the scaled error');
% Overlay the theoretical Normal density
plot(x_theoretical,f_theoretical,'-', ...
    'Color',[0.850 0.325 0.098], ...
    'DisplayName','Theoretical Normal density');
title('Fig 4. Asymptotic normality of the root-n scaled error');
xlabel(['Root-n scaled deviation of the sample proportion from ' ...
    'the true probability']);
ylabel('Probability density');
legend('show');
hold off
