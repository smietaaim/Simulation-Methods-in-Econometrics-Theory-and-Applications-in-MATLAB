% Exercise - Understanding sample-selection bias using simulation

%% 1. Aim of the exercise

% To understand how sample selection generates bias in OLS estimation using simulation.

%% 2. Initialize MATLAB environment

% Clear workspace, command window, and open figures
clear;
clc;
close all;

%% 3. Set data-generating process parameters

% 3.1. Number of observations
N_obs = 10000;

% 3.2. True regression coefficients
B_0 = 0.2;
B_1 = 0.5;

%% 4. Generate the sampling distribution of the OLS estimator

% 4.1. Number of Monte Carlo simulations
N_sim = 1000;

% 4.2. Preallocate storage for OLS estimates
B_hat_1_sim = NaN(N_sim,1);

% 4.3. Generate samples and estimate OLS repeatedly
for i = 1:N_sim
    % Generate explanatory variable
    x_0 = ones(N_obs,1);
    x_1 = random('Normal',0,1,[N_obs 1]);
    % Generate error term
    u = random('Normal',0,1,[N_obs 1]);
    % DGP of the population model
    y = B_0 * x_0 + B_1 * x_1 + u;
    % Sample-selection mechanism: observations selected based on x_1 and u
    selected = (x_1 + u > 0);
    y_obs = y(selected);
    x_0_obs = x_0(selected);
    x_1_obs = x_1(selected);
    % OLS estimation
    X = [x_0_obs,x_1_obs];
    % Obtain OLS statistics
    LSS = lss(y_obs,X);
    % Obtain the OLS estimate of beta
    B_hat_1 = LSS.B_hat(2,1);
    % Store estimate
    B_hat_1_sim(i,1) = B_hat_1;
end

%% 5. Bias of the OLS estimator

% 5.1. Compute the mean of the OLS estimates
B_hat_1_mean = mean(B_hat_1_sim);

% 5.2. Compute the bias of the OLS estimator
bias = B_hat_1_mean - B_1;

%% 6. Plot the sampling distribution of the OLS estimator

% Create a figure
figure('Position',[100 100 1000 1000]);
% Estimate the density
[f,x_grid] = ksdensity(B_hat_1_sim);
% Plot the density
plot(x_grid,f,...
     'Color',[0.000 0.000 0.000], ...
     'DisplayName','Sampling distribution of Beta\_hat\_1')
% Expand the x-axis range
xlim([min(x_grid) - 0.1 B_1 + 0.1])
hold on
% Add the mean of the sampling distribution
line([B_hat_1_mean B_hat_1_mean],ylim,...
     'Color',[1.000 0.000 0.000],...
     'DisplayName','Mean of Beta\_hat\_1')
% Add the true value
line([B_1 B_1],ylim,...
     'Color',[0.000 0.000 1.000],...
     'DisplayName','True value of Beta')
title(['Fig. 1. Sampling distribution of the OLS estimator ' ...
    'under sample selection'])
legend('show')
xlabel('Beta\_hat\_1')
ylabel('Density')
hold off
