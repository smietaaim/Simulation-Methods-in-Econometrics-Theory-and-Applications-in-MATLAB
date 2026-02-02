% Exercise - Understanding the consistency of the OLS estimator using simulation

%% 1. Aim of the exercise
% To understand the consistency of the OLS eatimator.

%% 2. Theory
% Refer to the accompanying PDF file for the theory.

%% 3. Application

% 3.1. Clear the memory
clear;

% 3.2. Set the number of simulations 
N_sim = 1000;

% 3.3. Set the sample size
N_obs = [1000 10000 100000];
N_obs_j = size(N_obs,2);

% 3.4. Set true values for the coefficients
B_true = [0.2 0.5]';

% 3.5. Define the number of coefficients to be simulated
N_par = 1;

% 3.6. Preallocate matrices for storing the simulated statistics 
B_hat_1_sim = NaN(N_sim,N_par);
B_hat_1_sim_j = NaN(N_sim,N_obs_j);

% 3.7. Create sampling distributions for the OLS estimator 
for j = 1:N_obs_j
    for i = 1:N_sim 
        u = random('Normal',0,1,[N_obs(1,j) 1]);
        x_0 = ones(N_obs(1,j),1);
        x_1 = random('Uniform',-1,1,[N_obs(1,j) 1]);
        X = [x_0 x_1];
        y = X*B_true+u;
        LSS = exercisefunctionlss(y,X);
        B_hat_1_sim(i,1) = LSS.B_hat(2,1);
        B_hat_1_sim_j(:,j) = B_hat_1_sim(:,1);
    end
end

%% 4. Plot the sampling distributions of the OLS estimator 
hold on
ksdensity(B_hat_1_sim_j(:,1))
ksdensity(B_hat_1_sim_j(:,2))
ksdensity(B_hat_1_sim_j(:,3))
title(['Fig. 1. Sampling distribution of the OLS estimator ' ...
    'and sample size'])
legend('N\_obs = 1,000','N\_obs = 10,000','N\_obs = 100,000')
ylabel('Frequency')
xlabel('B\_hat\_1') 
hold off
