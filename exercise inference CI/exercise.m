% Exercise - Understanding the confidence interval (CI) using simulation

%% 1. Aim of the exercise
% To learn how to interpret the CI.

%% 2. Theory
% Refer to the accompanying PDF file for the theory.

%% 3. Application

% 3.1. Clear the memory
clear;

% 3.2. Set the sample size
N_obs = 1000;

% 3.3. Generate data for the only independent variable
X = random('Uniform',-1,1,[N_obs 1]); 

% 3.4. Define the number of coefficients to be estimated
N_par = size(X,2);

% 3.5. Set a (hypothesized) value for the population coefficient
B_true = 0.5;

% 3.6. Set the number of simulations
N_sim = 300;

% 3.7. Preallocate a matrix for storing OLS estimates from all samples
B_hat_sim = NaN(N_sim,N_par);  

% 3.8. Preallocate a matrix for storing SSEs from repeated samples 
B_hat_SEE_sim = NaN(N_sim,N_par);

% 3.9. Coefficient and standard error estimates (SEE)
for i = 1:N_sim
    u = random('Normal',0,1,[N_obs 1]);
    y = X*B_true+u; % The data generating process (DGP)
    LSS = lss(y,X);
    B_hat_sim(i,1) = LSS.B_hat(1,1); % B_hat is a random variable
    B_hat_SEE_sim(i,1) = LSS.B_hat_SEE(1,1);
end 

%% 4. Construct random intervals (RI) using repeated samples

% 4.1. Define the significance level
alpha = 0.05; % For 95% CI. Change to 0.10 for 90% CI.

% 4.2. Calculate the degrees of freedom for the t distribution
nu = N_obs-N_par;

% 4.3. Calculate the critical value from the t distribution
t_c = tinv(1-alpha/2,nu);
 
% 4.4. Construct the RIs for B_true using its estimates from all samples
RIs = [B_hat_sim-t_c*B_hat_SEE_sim,B_hat_sim+t_c*B_hat_SEE_sim];

% 4.5. Construct the CI for B_true when there is one sample available
CI = RIs(1,:);

%% 5. Plot the RIs from all samples and the population coefficient

% 5.1. Plot the RIs from all samples and the population coefficient
figure('Position',[100 100 1000 1000]);
hold on
for i = 1:N_sim
    plot(RIs(i,:),[i,i],'k-','LineWidth',0.5);
    plot(B_true,i,'bo','MarkerSize',1.5,'MarkerFaceColor','b');
end
title('Fig. 2. Interval estimates');
ylabel('Sample index');
xlabel('Interval bound');
hold off

%% 6. Interpret the CI

% 6.1. Create a dummy indicating if B_true is within the RIs
B_true_is_within_RIs = B_true > RIs(:,1) & B_true < RIs(:,2);

% 6.2. Calculate the proportion of times B_true is within the RIs
Proportion_B_true_is_within_RIs = mean(B_true_is_within_RIs); % App. 0.95
