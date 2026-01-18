% Exercise - Understanding multicollinearity using simulation

%% 1. Aim of the exercise
% To learn the implications of multicollinearity for the OLS estimator.

%% 2. Theory
% Refer to the accompanying PDF file for the theory.

%% 3. Application

% 3.1. Clear the memory
clear;

% 3.2. Set the number of simulations 
N_sim = 1000;

% 3.3. Set the sample size
N_obs = 2000;

% 3.4. Set true values for the coefficients
B_true = [0.5 0.75]';

% 3.5. Create the constant term
x_0 = ones(N_obs,1);

% 3.6. Create a vector of covariances between two independent variables
sigma_x_1_x_2 = 0:0.11:0.99;
N_sig = size(sigma_x_1_x_2,2);

% 3.7. Define the number of coefficients to be simulated
N_par = 1;

% 3.8. Preallocate matrices for storing statistics to be simulated 
B_hat_1_sim = NaN(N_sim,N_par);
B_hat_1_sim_j = NaN(N_sim,N_sig);
B_hat_1_SE_sim = NaN(N_sim,N_par); 
B_hat_1_SE_sim_j = NaN(N_sim,N_sig); 
B_hat_1_sim_j_SD = NaN(N_sig,N_par); 

% 3.9. Define mean vector for the multivariate random number generator 
mu = [0 0];

% 3.10. Create the sampling distribution of the OLS esimator
for j = 1:N_sig
    for i = 1:N_sim
        Sigma = reshape([1 sigma_x_1_x_2(:,j) sigma_x_1_x_2(:,j) 1], ...
            2,2);
        x_1_x_2_mvn = mvnrnd(mu,Sigma,N_obs);
        x_1 = x_1_x_2_mvn(:,1);
        x_2 = x_1_x_2_mvn(:,2);
        X = [x_1 x_2];
        u = random('Normal',0,1,[N_obs 1]);
        y = X*B_true+u;
        LSS = exercisefunctionlss(y,X);
        B_hat_1_sim(i,1) = LSS.B_hat(1,1);
        B_hat_1_sim_j(:,j) = B_hat_1_sim(:,1);
        B_hat_1_SE_sim(i,1) = LSS.B_hat_SEE(2,1); 
        B_hat_1_SE_sim_j(:,j) = B_hat_1_SE_sim(:,1); 
        B_hat_1_sim_j_SD(j,:) = std(B_hat_1_sim_j(:,j));
    end
end

%% 4. Plot the sampling distribution of the OLS estimator
figure
hold on 
ksdensity(B_hat_1_sim_j(:,1))
ksdensity(B_hat_1_sim_j(:,10))
line([mean(B_hat_1_sim_j(:,1)) mean(B_hat_1_sim_j(:,1))],ylim, ...
    'Color','black')
line([mean(B_hat_1_sim_j(:,9)) mean(B_hat_1_sim_j(:,10))],ylim, ...
    'Color','black')
title(['Fig. 1: The Effect of multicollinearity on the sampling ' ...
    'distribution of the OLS estimator'])
legend('No correlation','Almost perfect correlation', ...
    'B\_hat\_1\_sim\_mean')
ylabel('Density')
xlabel('B\_hat\_1') 
hold off 

%% 5. Plot the SD of the sampling distribution of the OLS estimator
figure
hold on
plot(sigma_x_1_x_2,B_hat_1_sim_j_SD)
ylim([0 0.25])
title(['Fig. 2. Standard deviation of the sampling distribution ' ...
    'of the OLS estimator at different correlation Levels between ' ...
    'the independent variables'])
ylabel('S.D. of B\_hat\_1')
xlabel('Correlation between the indepdent variables') 
hold off

%% 6. Plot the sampling distribution of the SE estimator 
figure
hold on 
ksdensity(B_hat_1_SE_sim_j(:,1))
ksdensity(B_hat_1_SE_sim_j(:,10))
title(['Fig. 3. The effect of multicollinearity on the estimator ' ...
    'of the SE of the OLS Estimator'])
legend('No correlation','Almost perfect correlation')
ylabel('Density')
xlabel('B\_hat\_1\_SE') 
hold off
