% Exercise - Understanding serial correlation using simulation

%% 1. Aim of the exercise
% To understand the implications of serial correlaiton for the sampling 
% distribution of the OLS estimator using simulation.

%% 2. Theory
% Refer to the accompanying PDF file for the theory.

%% 3. Application

% 3.1. Clear the memory
clear;

% 3.2. Set the number of simulations 
N_sim = 1000;

% 3.3. Set the sample size
N_obs = 50;

% 3.4. Define the number of parameters to be simulated
N_par = 3;

% 3.5. Set true values for the coefficients of the model
B_true = [0.2 0.5]';

% 3.6. Generate data for the systematic component of the model
x_0 = ones(N_obs,1);
x_1 = random('Uniform',-1,1,[N_obs 1]);

% 3.7. Preallocate empty matrices for storing simulated coefficients 
B_hat_1_sim = NaN(N_sim,N_par);

% 3.8. Create the sampling distribution of the OLS estimator
for i = 1:N_sim 
    model = arima('Constant',0,'AR',0.00,'MA',0,'Distribution', ...
        'Gaussian','Variance',1); 
    u = simulate(model,N_obs); 
    X = [x_0 x_1];
    y = X*B_true+u;
    LSS = exercisefunctionlss(y,X);
    B_hat_1_sim(i,1) = LSS.B_hat(2,1);
    model = arima('Constant',0,'AR',0.85,'MA',0,'Distribution', ...
        'Gaussian','Variance',1); 
    u_ar = simulate(model,N_obs); 
    X = [x_0 x_1];
    y_ar = X*B_true+u_ar;
    LSS = exercisefunctionlss(y_ar,X);
    B_hat_1_sim(i,2) = LSS.B_hat(2,1);
    y_ar_lag = lagmatrix(y_ar,1); 
    X = [x_0 x_1 y_ar_lag];
    LSS = exercisefunctionlss(y_ar(2:N_obs,:),X(2:N_obs,:)); 
    B_hat_1_sim(i,3) = LSS.B_hat(2,1);
end

%% 4. Plot normal and serially correlated errors
figure
hold on 
plot(u)
plot(u_ar)
title('Fig. 1. Errors that are normal and serially correlated')
legend('Errors are normal','Errors are serially correlated')
ylabel('u')
xlabel('Random draws') 
hold off

%% 5. Sampling distribution of the OLS estimator and serial correlation
figure
hold on
ksdensity(B_hat_1_sim(:,1))
ksdensity(B_hat_1_sim(:,2))
line([mean(B_hat_1_sim(:,2)) mean(B_hat_1_sim(:,2))],ylim, ...
    'Color','blue')
line([mean(B_hat_1_sim(:,3)) mean(B_hat_1_sim(:,3))],ylim, ...
    'Color','red')
title('Fig. 2. Sampling distribution of the OLS estimator')
legend('Errors are normal','Errors are serially correlated')
ylabel('Kernel smoothed density')
xlabel('B\_hat\_1\_sim') 
hold off

%% 6. Sampling distribution of the OLS estimator and lagged dependent 
figure
hold on
ksdensity(B_hat_1_sim(:,2))
ksdensity(B_hat_1_sim(:,3))
line([mean(B_hat_1_sim(:,2)) mean(B_hat_1_sim(:,2))],ylim, ...
    'Color','blue')
line([mean(B_hat_1_sim(:,3)) mean(B_hat_1_sim(:,3))],ylim, ...
    'Color','red')
title('Fig. 3. Sampling distribution of the OLS estimator')
legend('Lagged dependent not included','Lagged dependent included', ...
    'Mean of B\_hat\_1\_sim when lagged dependent not included', ...
    'Mean of B\_hat\_1\_sim when lagged dependent included')
ylabel('Kernel smoothed density')
xlabel('B\_hat\_1\_sim') 
hold off
