% Exercise – Understanding indirect inference

%% 1. Aim of the exercise
% Demonstrate indirect inference by estimating the parameter of a moving
% average process (MA(1)) using an auxiliary autoregressive model (AR(1)).

%% 2. Theory
% Refer to the accompanying PDF file for theoretical background.

%% 3. Model setup
% Refer to the PDF for the setup of the complex and the auxiliary models.

%% 4. Set simulation parameters

% 4.1. Clear workspace
clear;

% 4.2. Number of simulated datasets
N_sim = 100;

% 4.3. True MA(1) parameter (structural)
theta_true = 0.5;

% 4.4. Standard deviation of error term
sigma_true = 1;

% 4.5. Length of time series
T = 1000;

% 4.6. Weight matrix for objective function (scalar)
W = 1;

%% 5. Simulate MA(1) process

% 5.1. Generate error terms
epsilon = random("Normal",0,sigma_true,[T+1,1]);

% 5.2. Initialize time series
y = NaN(T,1);

% 5.3. Generate MA(1) data using theta_true
for t = 2:T+1
    y(t-1) = epsilon(t)+theta_true*epsilon(t-1);
end

%% 6. Plot simulated data

% Plot
figure('Position',[100 100 1000 1000]);
plot(y)
title('Fig. 1. Simulated MA(1) process');
xlabel('Time');
ylabel('y_t');

%% 7. Define auxiliary model estimation function
% See the function file auxiliary.m.

%% 8. Estimate auxiliary model on observed data

% AR(1) estimate from MA(1) data
beta_hat = auxiliary(y); 

%% 9. Define indirect inference objective function
% See the function file objective.m.

%% 10. Estimate the parameter of the complex model using ind. inf.

% 10.1. Initial guess for optimal theta
theta_initial = 0.5;

% 10.2. Optimization output
options = optimoptions('fmincon', ...
    'Algorithm','sqp', ...
    'Diagnostics','off', ...
    'Display','iter-detailed', ...
    'MaxIterations',100, ...
    'MaxFunctionEvaluations',100, ...
    'OptimalityTolerance',1e-4, ...
    'StepTolerance',1e-4, ...
    'SpecifyObjectiveGradient',false);

% 10.3. Optimize to find theta that minimizes the objective function Q
theta_estimated = fmincon(@(theta) ...
    objective(theta,N_sim,T,W,beta_hat),theta_initial, ...
    [],[],[],[],[],[],[],options);

%% 11. Simulated estimation of AR(1) coefficients via MA(1) model

% 11.1. Preallocate vector to store AR(1) estimates from simulations
beta_hat_sim = NaN(N_sim,1);

% 11.2. Simulate data using estimated theta and re-estimate AR(1)
for i = 1:N_sim
    epsilon_sim = random("Normal",0,1,[T+1,1]);
    y_sim = zeros(T,1);
    for t = 2:T+1
        y_sim(t-1) = epsilon_sim(t)+theta_estimated*epsilon_sim(t-1);
    end
    beta_hat_sim(i) = auxiliary(y_sim);
end

%% 12. Compute Wald statistic and critical value

% 12.1. Variance of simulated AR(1) estimates
var_beta_hat = var(beta_hat_sim);

% 12.2. Wald statistic
wald_statistic = (theta_estimated-theta_true)^2/var_beta_hat;

% 12.3. Display Wald statistic
disp(['Wald statistic: ',num2str(wald_statistic)]);

% 12.4. Degrees of freedom
df = 1;

% 12.5. Critical value from chi-squared distribution
chisq_critical_value = chi2inv(0.95,df);

% 12.6. Hypothesis test
if wald_statistic > chisq_critical_value
    disp(['Reject the null hypothesis: ' ...
        'The parameters are significantly different.']);
else
    disp(['Fail to reject the null hypothesis: ' ...
        'The parameters are not significantly different.']);
end

%% 13. Simulation stability and the shape of the objective function

% 13.1. Range of candidate theta values
theta_grid = linspace(0,1,100);

% 13.2. Preallocate for objective values
Q_values = zeros(size(theta_grid));

% 13.3. Compute the AR(1) estimate from original MA(1) data
beta_hat = auxiliary(y);

% 13.4. Loop over theta candidates
for i = 1:length(theta_grid)
    Q_values(i) = objective(theta_grid(i),N_sim,T,W,beta_hat);
end

%% 14. Plot

% Plot
figure('Position',[100 100 1000 1000]);
plot(theta_grid,Q_values);
title('Fig. 2. Objective function Q(\theta) vs. candidate \theta');
xlabel('\theta (candidate)');
ylabel('Q(\theta)');
