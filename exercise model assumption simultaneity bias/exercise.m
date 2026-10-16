% Exercise - Understanding the simultaneity bias using simulation

%% 1. Aim of the exercise

% To understand why simultaneity creates endogeneity.

%% 2. Theory

% Refer to the accompanying PDF file for the theory.

%% 3. Clear memory

% Clear the memory
clear;
clc;
close all

%% 4. Simulating a simultaneous equations model

% 4.1. Set sample size
N_obs = 1000;

% 4.2. Set structural parameters
B_1 = 0.5;
B_2 = 0.6;
g_1 = 1.0;
g_2 = 1.0;

% 4.3. Generate exogenous variables
z_1 = random('Normal',0,1,[N_obs 1]);
z_2 = random('Normal',0,1,[N_obs 1]);

% 4.4. Generate structural errors
u_1 = random('Normal',0,1,[N_obs 1]);
u_2 = random('Normal',0,1,[N_obs 1]);

% 4.5. Solve the simultaneous system
y_1 = (g_1 * z_1 + B_1 * g_2 * z_2 + u_1 + B_1 * u_2) / ...
    (1 - B_1 * B_2);
y_2 = (B_2 * g_1 * z_1 + g_2 * z_2 + B_2 * u_1 + u_2) / ...
    (1 - B_1 * B_2);

% 4.6. Compute the correlation between y_2 and u_1
corr_y_2_u_1 = corr(y_2,u_1);

%% 5. OLS estimation of the first structural equation

% Create the systematic part of the regression
X = [y_2 z_1];

%% 6. Sampling distribution of the OLS estimator

% 6.1. Set the number of simulations
N_sim = 1000;

% 6.2. Preallocate a vector for the simulated estimates of B_1
B_hat_1_sim = NaN(N_sim,1);

% 6.3. Create the sampling distribution
for i = 1:N_sim
    z_1 = random('Normal',0,1,[N_obs 1]);
    z_2 = random('Normal',0,1,[N_obs 1]);
    u_1 = random('Normal',0,1,[N_obs 1]);
    u_2 = random('Normal',0,1,[N_obs 1]);
    y_1 = (g_1 * z_1 + B_1 * g_2 * z_2 + u_1 + B_1 * u_2) / ...
        (1 - B_1 * B_2);
    y_2 = (B_2 * g_1 * z_1 + g_2 * z_2 + B_2 * u_1 + u_2) / ...
        (1 - B_1 * B_2);
    X = [y_2 z_1];
    LSS = lss(y_1,X);
    B_hat_1_sim(i,1) = LSS.B_hat(1,1);
end

%% 7. Bias of the OLS estimator

% 7.1. Compute the mean of the sampling distribution
B_hat_1_mean = mean(B_hat_1_sim);

% 7.2. Compute the bias
bias = B_hat_1_mean - B_1;

%% 8. Plot the sampling distribution of the OLS estimator

% Create a figure
figure('Position',[100 100 1000 1000]);
% Estimate the density
[f,x] = ksdensity(B_hat_1_sim);
% Plot the density
plot(x,f,...
    'Color',[0.000 0.000 0.000],...
    'DisplayName','Sampling distribution of B\_hat\_1')
% Expand the x-axis range
lower = min([x(:);B_1]) - 0.1;
upper = max([x(:);B_1]) + 0.1;
xlim([lower upper])
hold on
% Add the mean of the sampling distribution
line([B_hat_1_mean B_hat_1_mean],ylim,...
    'Color',[1.000 0.000 0.000],...
    'DisplayName','Mean of B\_hat\_1')
% Add the true value
line([B_1 B_1],ylim,...
    'Color',[0.000 0.000 1.000],...
    'DisplayName','True value of B\_1')
title('Fig. 1. Simultaneity bias in the OLS estimator')
legend('show')
ylabel('Density')
xlabel('B\_hat\_1')
hold off

%% 9. Understanding when simultaneity bias disappears

% 9.1. Set B_2 equal to zero
% B_2 = 0.0;

% 9.2. Repeat the simulation exercise
% Re-run Sections 3 through 7 and compare the results.

% 9.3. Compare the two sampling distributions
% Is the mean of B_1_hat_sim now closer to the true value B_1?

% 9.4. Explain the result
% Does y_2 still contain u_1 when B_2 equals zero?
