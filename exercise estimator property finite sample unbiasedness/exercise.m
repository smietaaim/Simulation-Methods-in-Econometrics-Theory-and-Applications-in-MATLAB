% Exercise - Understanding the unbiasedness property of the OLS estimator 

%% 1. Aim of the exercise

% The aim of this exercise is to understand the unbiasedness property of
% the OLS estimator.

%% 2. Clear the workspace   

% Clear the workspace
clear;

%% 3. Setup for the Monte Carlo simulation

% 3.1. Define the sample size
N_obs = 1000;

% 3.2. Define the number of repeated samples from the population
N_sim = 1000;

% 3.3. Generate data for the independent variable
X = random('Uniform',-1,1,[N_obs 1]); 

% 3.4. Define the population coefficient
B_true = 0.5;

%% 4. Create the sampling distribution of the OLS estimator

% 4.1. Preallocate matrix to store OLS estimates from repeated samples
B_hat_sim = NaN(1,N_sim);

% 4.2. Conduct the Monte Carlo simulation
for i = 1:N_sim
    u = random('Normal',0,1,[N_obs 1]);
    y = X*B_true+u;
    LSS = exercisefunctionlss(y,X);
    B_hat_sim(1,i) = LSS.B_hat(1,1);
end 

%% 5. Plot the sampling distribution of the OLS estimator as a density

% Figure
figure
hold on
[f,xi] = ksdensity(B_hat_sim(1,:));
plot(xi,f, ...
    'DisplayName','Sampling distribution of B\_hat');
line([mean(B_hat_sim(1,:)) mean(B_hat_sim(1,:))],ylim, ...
    'DisplayName','Mean of B\_hat');
line([B_true B_true],ylim, ...
     'DisplayName','True B');
title('Fig. 1. Sampling distribution of the OLS estimator and its mean')
xlabel('B\_hat')
ylabel('Density')
legend('show')
hold off
