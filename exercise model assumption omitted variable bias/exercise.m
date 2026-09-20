% Exercise - Understanding the omitted variable bias using simulation

%% 1. Aim of the exercise

% To learn how an omitted variable leads to a biased OLS estimate.

%% 2. Theory

% Refer to the accompanying PDF file for the theory.

%% 3. Application

% 3.1. Clear the memory 
clear;

% 3.2. Set the number of simulations 
N_sim = 1000;

% 3.3. Set the sample size
N_obs = 1000;

% 3.4. Set true values for the coefficients
B_true = [0.5 0.75]';

% 3.5. Create the constant term
x_0 = ones(N_obs,1);

% 3.6. Create a vector of covariances between two independent variables
sigma_x_1_x_2 = 0:0.11:0.99;

% 3.7. Number of covariance scenarios
N_sig = size(sigma_x_1_x_2,2);

% 3.8. Preallocate a matrix for storing OLS estimates from all samples
B_hat_1_sim = NaN(N_sim,N_sig);

% 3.9. Define mean vector for the multivariate random number generator 
mu = [0 0];

% 3.10. Create the sampling distribution of the biased OLS estimator
for j = 1:N_sig
    for i = 1:N_sim
        Sigma = reshape([1 sigma_x_1_x_2(:,j) sigma_x_1_x_2(:,j) 1] ...
            ,2,2); 
        x_1_x_2_mvn = mvnrnd(mu,Sigma,N_obs);
        x_1 = x_1_x_2_mvn(:,1);
        x_2 = x_1_x_2_mvn(:,2);
        X = [x_1 x_2];
        u = random('Normal',0,1,[N_obs 1]);
        y = X*B_true+u;
        LSS = exercisefunctionlss(y,x_1); 
        B_hat_1_sim(i,j) = LSS.B_hat(1,1); 
    end
end

%% 4. Plot the sampling distribution of the biased OLS estimator

% Plot the figure
hold on
[f1,x1] = ksdensity(B_hat_1_sim(:,1));
plot(x1,f1, ...
    'Color',[1.0 1.0 0.0])
[f2,x2] = ksdensity(B_hat_1_sim(:,10));
plot(x2,f2, ...
    'Color',[1.0 0.0 0.0])
line([mean(B_hat_1_sim(:,1)) mean(B_hat_1_sim(:,1))],ylim, ...
    'Color',[1.0 1.0 0.0])
line([mean(B_hat_1_sim(:,10)) mean(B_hat_1_sim(:,10))],ylim, ...
    'Color',[1.0 0.0 0.0])
line([B_true(1,1) B_true(1,1)],ylim, ...
    'Color','black')
title(['Fig. 1. The effect of omitting a variable on the distribution ' ...
    'of the OLS estimator'])
legend('Correlation between x_1 and x_2 is 0', ...
    'Correlation between x_1 and x_2 is almost perfect', ...
    ['B\_hat\_1\_sim\_mean ' ...
    'when correlation between x_1 and x_2 is 0'], ...
    ['B\_hat\_1\_sim\_mean ' ...
    'when correlation between x_1 and x_2 is almost perfect'], ...
    'B\_true')
ylabel('Density')
xlabel('B\_hat\_1')
xticks(0.2:0.1:1.4)
hold off
