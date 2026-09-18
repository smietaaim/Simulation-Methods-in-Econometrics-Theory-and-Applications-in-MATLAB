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

% 2.4. Set population coefficients
B_true_0 = 0.2;
B_true_1 = 0.5;
B_true = [B_true_0; B_true_1];

% 2.5. Set the number of simulations 
N_sim = 1000;

% 2.6. Preallocate a matrix for storing OLS estimates from all samples
B_hat_0_sim = NaN(1,N_sim);
B_hat_1_sim = NaN(1,N_sim);

% 2.7. Preallocate a matrix for storing the SEEs from repeated samples 
B_hat_0_SEE_sim = NaN(1,N_sim);
B_hat_1_SEE_sim = NaN(1,N_sim);

% 2.8. Create the sampling distribution of the OLS estimator
for i = 1:N_sim
    % Draw new error for each sample in each iteration of the loop
    u = random('Normal',0,1,[N_obs 1]);
    % Generate values for the dependent variable
    y = X*B_true+u; % DGP
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

% Figure
figure;
set(gcf,'Position',[100 100 1000 1000]); 
hold on
histogram(B_hat_1_sim(1,:),50)
title('Fig. 1. Sampling distribution of the OLS estimator')
legend('Sampling distribution of B\_hat\_1 based on Monte Carlo sim',...
    'B\_hat\_1\_sim\_mean')
ylabel('Frequency')
xlabel('B\_hat\_1')
hold off

%% 4. Plot the sampling distribution of the OLS estimator as a density

% Figure
figure;
set(gcf,'Position',[100 100 1000 1000]); 
hold on
ksdensity(B_hat_1_sim(1,:))
title('Fig. 2. Sampling distribution of the OLS estimator')
legend('Sampling distribution of B\_hat\_1 based on Monte Carlo sim',...
    'B\_hat\_1\_sim\_mean')
ylabel('Density')
xlabel('B\_hat\_1')
hold off 

%% 5. SE of a statistic is the SD of its sampling distribution

% 5.1. Calculate the standard deviation of the sampling distribution
std(B_hat_1_sim(1,:))

% 5.2. Calculate the standard error of the sampling distribution
LSS.B_hat_SEE(2,1)

%% 6. Plot the sampling distribution of the SE estimator

% Figure
figure;
set(gcf,'Position',[100 100 1000 1000]); 
hold on
ksdensity(B_hat_1_SEE_sim(1,:))
title(['Fig. 3. Sampling distribution of the estimator of the ...' ...
    'SE of the OLS estimator'])
legend(['Sampling distribution of the SE estimator of ...' ...
    'B\_hat\_1 based on Monte Carlo sim'],'B\_hat\_1\_SEE\_sim\_mean')
ylabel('Frequency')
xlabel('B\_hat\_1\_SEE')
hold off
