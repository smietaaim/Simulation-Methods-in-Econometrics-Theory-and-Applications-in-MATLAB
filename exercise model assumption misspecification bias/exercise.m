% Exercise - Understanding misspecification bias using simulation

%% 1. Aim of the exercise

% To learn how model misspecification can lead to a biased OLS estimate.

%% 2. Theory

% Refer to the accompanying PDF file for the theory.

%% 3. Initialize MATLAB environment

% Clear workspace, command window, and open figures
clear;
clc;
close all;

%% 4. Set true data-generating process parameters

% 4.1. Number of observations
N_obs = 250;

% 4.2. True parameters of the nonlinear population model
a_0 = 1.0;
a_1 = 1.5;

%% 5. Create the true data-generating process

% 5.1. Generate the constant term
x_0 = ones(N_obs,1);

% 5.2. Generate random values from the uniform distribution
x_1 = random('Uniform',0,2,[N_obs,1]);

% 5.3. Generate random values from the normal distribution
u = random('Normal',0,1,[N_obs,1]);

% 5.4. DGP of the population model
y = a_0 * x_0 + exp(a_1 * x_1) + u;

%% 6. Obtain OLS statistics

% 6.1. Generate regressor vector
X = [x_0 x_1];

% 6.2. Assume an incorrect linear model and estimate it with OLS
LSS = lss(y,X);

% 6.3. Obtain fitted values
y_hat = LSS.y_hat(:,:);

%% 7. Plot the scatter diagram and the OLS fitted line

% 7.1. Sort for a smooth fitted line
[x_1_sorted,idx] = sort(x_1);
y_hat_sorted = y_hat(idx);

% 7.2. Figure
figure('Position',[100 100 1000 1000]);
scatter(x_1,y,...
    'filled','black',...
    'DisplayName','Scatter plot')
hold on
plot(x_1_sorted,y_hat_sorted,...
    'Color','blue',...
    'DisplayName','OLS fitted line')
hold off
title('Fig. 1. Nonlinear Population Model and Linear OLS Fit')
legend('show')

%% 8. Demonstrate model misspecification

% 8.1. Obtain the residuals
u_hat = LSS.u_hat(:,:);

% 8.2. Plot residuals against the regressor
figure('Position',[100 100 1000 1000]);
scatter(x_1,u_hat,...
    'filled',...
    'DisplayName','Residuals')
hold on
yline(0,...
    'r',...
    'DisplayName','Zero residual')
hold off
title('Fig. 2. Residuals versus Regressor')
legend('show')

%% 9. Set Monte Carlo simulation parameters

% 9.1. Number of Monte Carlo replications
N_sim = 1000;

% 9.2. Store OLS slope estimates
B_hat_1_sim = NaN(N_sim,1);

%% 10. Conduct the Monte Carlo simulation

% Create the sampling distribution
for i = 1:N_sim
    % Generate the constant term
    x_0 = ones(N_obs,1);
    % Generate the regressor
    x_1 = random('Uniform',0,2,[N_obs,1]);
    % Generate the disturbance term
    u = random('Normal',0,1,[N_obs,1]);
    % Generate data from the nonlinear population model
    y = a_0 * x_0 + exp(a_1 * x_1) + u;
    % Generate regressor vector
    X = [x_0 x_1];
    % Estimate an incorrect linear model
    LSS = lss(y,X);
    % Store the slope estimate
    B_hat_1_sim(i) = LSS.B_hat(2);
end

%% 11. Evaluate the sampling distribution

% 11.1. Mean OLS estimate
B_hat_1_sim_mean = mean(B_hat_1_sim);

% 11.2. Difference from the DGP parameter
bias = B_hat_1_sim_mean - a_1;

%% 12. Plot the sampling distribution

% Create a figure
figure('Position',[100 100 1000 1000]);
% Estimate the density
[f,x] = ksdensity(B_hat_1_sim);
% Plot the density
hold on
plot(x,f,...
     'Color',[0.000 0.000 0.000],...
     'DisplayName','Sampling distribution of B\_hat\_1')
% Expand the x-axis range
lower = min([x(:);a_1]) - 0.1;
upper = max([x(:);a_1]) + 0.1;
xlim([lower upper])
% Add the mean of the sampling distribution
line([B_hat_1_sim_mean B_hat_1_sim_mean],ylim,...
     'Color',[1.000 0.000 0.000],...
     'DisplayName','Mean of B\_hat\_1')
% Add the parameter used in the DGP
line([a_1 a_1],ylim,...
     'Color',[0.000 0.000 1.000],...
     'DisplayName','a\_1')
hold off
title(['Fig. 3. Sampling Distribution of the OLS Estimator ' ...
       'under Functional Form Misspecification'])
legend('show')
xlabel('B\_hat\_1')
ylabel('Density')
