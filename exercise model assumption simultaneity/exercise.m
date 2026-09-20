% Exercise - Understanding the simultaneity bias using simulation

%% 1. Aim of the exercise

% To understand why simultaneity creates endogeneity.

%% 2. Theory

% Refer to the accompanying PDF file for the theory.

%% 3. Application

% 3.1. Clear memory
clear;

% 3.2. Set sample size
N_obs = 1000;

% 3.3. Set structural parameters
alpha_1 = 0.5;
alpha_2 = 0.6;
beta_1 = 1.0;
beta_2 = 1.0;

% 3.4. Generate exogenous variables
z_1 = random('Normal',0,1,[N_obs 1]);
z_2 = random('Normal',0,1,[N_obs 1]);

% 3.5. Generate structural errors
u_1 = random('Normal',0,1,[N_obs 1]);
u_2 = random('Normal',0,1,[N_obs 1]);

% 3.6. Solve the simultaneous system
y_1 = (beta_1 * z_1 + alpha_1 * beta_2 * z_2 + ...
    u_1 + alpha_1 * u_2) / ...
    (1 - alpha_1 * alpha_2);

y_2 = (alpha_2 * beta_1 * z_1 + beta_2 * z_2 + ...
    alpha_2 * u_1 + u_2) / ...
    (1 - alpha_1 * alpha_2);

% 3.7. Compute the correlation between y_2 and u_1
corr_y_2_u_1 = corr(y_2,u_1);

%% 4. Estimate the first structural equation by OLS

% 4.1. Create the systematic part of the regression
X = [y_2 z_1];

% 4.2. Obtain OLS statistics
LSS = lss(y_1,X);

% 4.3. Obtain the OLS estimate of alpha_1
alpha_1_hat = LSS.B_hat(1,1);

%% 5. Create the sampling distribution of alpha_1_hat

% 5.1. Set the number of simulations
N_sim = 1000;

% 5.2. Preallocate a vector for the simulated estimates of alpha_1
alpha_1_hat_sim = NaN(N_sim,1);

% 5.3. Create the sampling distribution
for i = 1:N_sim
    z_1 = random('Normal',0,1,[N_obs 1]);
    z_2 = random('Normal',0,1,[N_obs 1]);
    u_1 = random('Normal',0,1,[N_obs 1]);
    u_2 = random('Normal',0,1,[N_obs 1]);
    y_1 = (beta_1 * z_1 + alpha_1 * beta_2 * z_2 + ...
           u_1 + alpha_1 * u_2) / ...
          (1 - alpha_1 * alpha_2);
    y_2 = (alpha_2 * beta_1 * z_1 + beta_2 * z_2 + ...
           alpha_2 * u_1 + u_2) / ...
          (1 - alpha_1 * alpha_2);
    X = [y_2 z_1];
    LSS = lss(y_1,X);
    alpha_1_hat_sim(i,1) = LSS.B_hat(1,1);
end

%% 6. Compute the bias of the OLS estimator

% 6.1. Compute the mean of the sampling distribution
alpha_1_hat_mean = mean(alpha_1_hat_sim);

% 6.2. Compute the bias
bias = alpha_1_hat_mean - alpha_1;

%% 7. Plot the sampling distribution of the OLS estimator

% Create a figure
figure;
% Set the figure size
set(gcf,'Position',[100 100 1000 1000]); 
% Estimate the density
[f,x] = ksdensity(alpha_1_hat_sim);
% Plot the density
plot(x,f, ...
    'Color',[0.000 0.000 0.000])
% Expand the x-axis range
xlim([alpha_1 - 0.1 max(x) + 0.1])
% Add the mean of the sampling distribution
hold on
line([alpha_1_hat_mean alpha_1_hat_mean],ylim, ...
    'Color',[1.000 0.000 0.000])
% Add the true value
line([alpha_1 alpha_1],ylim, ...
    'Color',[0.000 0.000 1.000])
% Add labels
title('Fig. 1. Simultaneity bias in the OLS estimator')
legend('Sampling distribution of alpha\_1\_hat', ...
    'Mean of alpha\_1\_hat', ...
    'True value of alpha\_1')
ylabel('Density')
xlabel('alpha\_1\_hat')
hold off

%% 8. Understanding when simultaneity bias disappears

% 8.1. Set alpha_2 equal to zero
% alpha_2 = 0.0;

% 8.2. Repeat the simulation exercise
% Hint: Re-run Sections 3 through 7 and compare the results.

% 8.3. Compare the two sampling distributions
% Hint: Is the mean of alpha_1_hat_sim now closer to the true value 
% alpha_1?

% 8.4. Explain the result
% Hint: Does y_2 still contain u_1 when alpha_2 equals zero?
