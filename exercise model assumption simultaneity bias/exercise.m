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
a_1 = 0.5;
a_2 = 0.6;
B_1 = 1.0;
B_2 = 1.0;

% 4.3. Generate exogenous variables
z_1 = random('Normal',0,1,[N_obs 1]);
z_2 = random('Normal',0,1,[N_obs 1]);

% 4.4. Generate structural errors
u_1 = random('Normal',0,1,[N_obs 1]);
u_2 = random('Normal',0,1,[N_obs 1]);

% 4.5. Solve the simultaneous system
y_1 = (B_1 * z_1 + a_1 * B_2 * z_2 + u_1 + a_1 * u_2) / ...
    (1 - a_1 * a_2);
y_2 = (a_2 * B_1 * z_1 + B_2 * z_2 + a_2 * u_1 + u_2) / ...
    (1 - a_1 * a_2);

% 4.6. Compute the correlation between y_2 and u_1
corr_y_2_u_1 = corr(y_2,u_1);

%% 5. OLS estimation of the first structural equation

% Create the systematic part of the regression
X = [y_2 z_1];

%% 6. Sampling distribution of the OLS estimator

% 6.1. Set the number of simulations
N_sim = 1000;

% 6.2. Preallocate a vector for the simulated estimates of a_1
a_hat_1_sim = NaN(N_sim,1);

% 6.3. Create the sampling distribution
for i = 1:N_sim
    z_1 = random('Normal',0,1,[N_obs 1]);
    z_2 = random('Normal',0,1,[N_obs 1]);
    u_1 = random('Normal',0,1,[N_obs 1]);
    u_2 = random('Normal',0,1,[N_obs 1]);
    y_1 = (B_1 * z_1 + a_1 * B_2 * z_2 + u_1 + a_1 * u_2) / ...
        (1 - a_1 * a_2);
    y_2 = (a_2 * B_1 * z_1 + B_2 * z_2 + a_2 * u_1 + u_2) / ...
        (1 - a_1 * a_2);
    X = [y_2 z_1];
    LSS = lss(y_1,X);
    a_hat_1_sim(i,1) = LSS.B_hat(1,1);
end

%% 7. Bias of the OLS estimator

% 7.1. Compute the mean of the sampling distribution
a_hat_1_mean = mean(a_hat_1_sim);

% 7.2. Compute the bias
bias = a_hat_1_mean - a_1;

%% 8. Plot the sampling distribution of the OLS estimator

% Create a figure
figure('Position',[100 100 1000 1000]);
% Estimate the density
[f,x] = ksdensity(a_hat_1_sim);
% Plot the density
plot(x,f,...
    'Color',[0.000 0.000 0.000],...
    'DisplayName','Sampling distribution of a\_hat\_1')
% Expand the x-axis range
lower = min([x(:);a_1]) - 0.1;
upper = max([x(:);a_1]) + 0.1;
xlim([lower upper])
hold on
% Add the mean of the sampling distribution
line([a_hat_1_mean a_hat_1_mean],ylim,...
    'Color',[1.000 0.000 0.000],...
    'DisplayName','Mean of a\_hat\_1')
% Add the true value
line([a_1 a_1],ylim,...
    'Color',[0.000 0.000 1.000],...
    'DisplayName','True value of a\_1')
title('Fig. 1. Simultaneity bias in the OLS estimator')
legend('show')
ylabel('Density')
xlabel('a\_hat\_1')
hold off

%% 9. Understanding when simultaneity bias disappears

% 9.1. Set a_2 equal to zero
% a_2 = 0.0;

% 9.2. Repeat the simulation exercise
% Re-run Sections 3 through 7 and compare the results.

% 9.3. Compare the two sampling distributions
% Is the mean of a_1_hat_sim now closer to the true value a_1?

% 9.4. Explain the result
% Does y_2 still contain u_1 when a_2 equals zero?
