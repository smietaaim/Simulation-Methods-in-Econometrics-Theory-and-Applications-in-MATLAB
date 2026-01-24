% Exercise - Sampling distribution of the OLS estimator 

%% 1. Aim of the exercise
% To learn the sampling distribution of the OLS estimator.

%% 2. Coefficient and standard error estimates (SEE) from repeated samples
 
% 2.1. Clear the memory
clear;

% 2.2. Set the sample size
N_obs = 9000;

% 2.3. Generate data for the independent variables
x_0 = ones(N_obs,1);
x_1 = random('Uniform',-1,1,[N_obs 1]); 
X = [x_0 x_1];

% 2.4. Set (hypothesized) values for the population coefficients
B_true_0 = 0.2;
B_true_1 = 0.5;
B_true = [B_true_0; B_true_1];

% 2.5. Set the number of simulations 
N_sim = 1000;

% 2.6. Preallocate a matrix for storing OLS estimates from all samples
B_hat_0_sim = NaN(1,N_sim);
B_hat_1_sim = NaN(1,N_sim);

% 2.7. Preallocate a matrix for storing the standard error estimates 
% (SSEs) from repeated samples 
B_hat_0_SEE_sim = NaN(1,N_sim);
B_hat_1_SEE_sim = NaN(1,N_sim);

% 2.8. Create the sampling distribution of the OLS estimator
for i = 1:N_sim
    % Draw new error for each sample (in each iteration of the loop)
    u = random('Normal',0,1,[N_obs 1]);
    % Generate values for the dependent variable
    y = X*B_true+u; % The data generating process (DGP)
    % Obtain OLS statistics using the external function
    LSS = exercisefunctionlss(y,X);
    % Store the OLS estimates
    B_hat_0_sim(1,i) = LSS.B_hat(1,1);
    B_hat_1_sim(1,i) = LSS.B_hat(2,1);
    % Store the SE estimates of OLS estimates 
    B_hat_0_SEE_sim(1,i) = LSS.B_hat_SEE(1,1);
    B_hat_1_SEE_sim(1,i) = LSS.B_hat_SEE(2,1);
end 

%% 3. Plot the sampling distribution of the OLS estimator  
figure
hold on
histogram(B_hat_1_sim(1,:),50)
line([mean(B_hat_1_sim(1,:)) mean(B_hat_1_sim(1,:))],ylim,'Color','red')
title('Fig. 1. Sampling distribution of the OLS estimator')
legend('Sampling distribution of B\_hat\_1 based on Monte Carlo sim',...
    'B\_hat\_1\_sim\_mean')
ylabel('Frequency')
xlabel('B\_hat\_1')
hold off

%% 4. Plot the sampling distribution of the OLS estimator as a density
figure
hold on
ksdensity(B_hat_1_sim(1,:))
line([mean(B_hat_1_sim(1,:)) mean(B_hat_1_sim(1,:))],ylim,'Color','red')
title('Fig. 2. Sampling distribution of the OLS estimator')
legend('Sampling distribution of B\_hat\_1 based on Monte Carlo sim',...
    'B\_hat\_1\_sim\_mean')
ylabel('Density')
xlabel('B\_hat\_1')
hold off 

%% 5. The mean of the sampling distribution of the OLS estimator
mean(B_hat_1_sim)

%% 6. SE of a statistic is the SD of its sampling distribution
std(B_hat_1_sim(1,:))
LSS.B_hat_SEE(2,1)

%% 7. Plot the sampling distribution of the SE estimator  
figure
hold on
ksdensity(B_hat_1_SEE_sim(1,:))
line([mean(B_hat_1_SEE_sim(1,:)) mean(B_hat_1_SEE_sim(1,:))],ylim,...
    'Color','red')
title(['Fig. 3. Sampling distribution of the estimator of the ...' ...
    'SE of the OLS estimator'])
legend(['Sampling distribution of the SE estimator of ...' ...
    'B\_hat\_1 based on Monte Carlo sim'],'B\_hat\_1\_SEE\_sim\_mean')
ylabel('Frequency')
xlabel('B\_hat\_1\_SEE')
hold off
