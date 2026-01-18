% Exercise - Understanding the normality assumption of the regression error

%% 1. Aim of the exercise 
% To understand how the OLS estimator behaves when the errors of the 
% regression are not normal.

%% 2. Application

% 2.1. Clear the memory
clear;

% 2.2. Set the number of simulations 
N_sim = 1000;

% 2.3. Set the sample size
N_obs = 350;

% 2.4. Set true values for the coefficients
B_true = [0.2 3.5]';

% 2.5. Define the number of coefficients to be simulated
N_par = 1;

% 2.6. Create the systematic component of the regression equation
x_0 = ones(N_obs,1);
x_1 = random('Uniform',-1,1,[N_obs 1]);
X = [x_0 x_1];

% 2.7. Preallocate matrices for storing statistics to be simulated 
B_hat_1_sim_OLS_normal = NaN(N_sim,N_par);
B_hat_1_sim_IRLS_normal = NaN(N_sim,N_par);
B_hat_1_sim_OLS_t = NaN(N_sim,N_par);
B_hat_1_sim_IRLS_t = NaN(N_sim,N_par);

% 2.8. Define the degrees of freedom for t distribution
t_df = 2;

% 2.9. Create sampling distributions for the OLS and IRLS estimators  
for i = 1:N_sim 
    u_normal = random('Normal',0,1,[N_obs 1]);
    y_normal = X*B_true+u_normal;
    u_t = random('t',t_df,[N_obs 1]);
    y_t = X*B_true+u_t;
    OLS = robustfit(x_1,y_normal,'ols');
    B_hat_1_sim_OLS_normal(i,1) = OLS(2,1);
    IRLS = robustfit(x_1,y_normal,'bisquare');
    B_hat_1_sim_IRLS_normal(i,1) = IRLS(2,1); 
    OLS = robustfit(x_1,y_t,'ols');
    B_hat_1_sim_OLS_t(i,1) = OLS(2,1);
    IRLS = robustfit(x_1,y_t,'bisquare');
    B_hat_1_sim_IRLS_t(i,1) = IRLS(2,1); 
end

%% 3. Plot errors with different distributional assumptions   
figure
hold on
ksdensity(u_normal)
ksdensity(u_t)
title(['Fig. 1. Distribution of regression errors with different ' ...
    'distributional assumptions'])
legend('Errors are standard normal','Errors are t')
ylabel('Density')
xlabel('u')
hold off

%% 4. Plot the sampling distributions of estimators when errors are normal 
figure
hold on
ksdensity(B_hat_1_sim_OLS_normal(:,1))
ksdensity(B_hat_1_sim_IRLS_normal(:,1))
title(['Fig. 2. Sampling distributions of the OLS and IRLS ' ...
    'estimators when errors are normal'])
legend('OLS estimator when errors are normal',['IRLS estimator ' ...
    'when errors are normal'])
ylabel('Density')
xlabel('u')
hold off

%% 5. Plot the sampling distributions of estimators when errors are t  
figure
hold on
ksdensity(B_hat_1_sim_OLS_t(:,1))
ksdensity(B_hat_1_sim_IRLS_t(:,1))
title(['Fig. 3. Sampling distributions of OLS and IRLS estimators ' ...
    'when errors are t'])
legend('OLS estimator when errors are t',['IRLS estimator ' ...
    'when errors are t'])
ylabel('Density')
xlabel('u')
hold off

%% 6. Plot the fitted models using the OLS and IRLS estimators
figure
hold on
scatter(x_1,y_t,'filled'); 
grid on; 
y_t_hat_OLS = OLS(1)+OLS(2)*x_1;
y_t_hat_IRLS = IRLS(1)+IRLS(2)*x_1; 
plot(x_1,y_t_hat_OLS,'red','LineWidth',2);
plot(x_1,y_t_hat_IRLS,'green','LineWidth',2)
title('Fig. 4. Fitted models using the OLS and IRLS estimators')
legend('Data','OLS regression','IRLS regression')
grid off;
ylabel('y\_t')
xlabel('x\_1')
hold off

%% 7. Other simulation experiments
figure
hold on
ksdensity(B_hat_1_sim_OLS_normal(:,1))
ksdensity(B_hat_1_sim_OLS_t(:,1))
title(['Fig. 5. Sampling distribution of the OLS estimator ' ...
    'when errors are normal and t'])
legend('OLS estimator when errors are normal',['OLS estimator ' ...
    'when errors are t'])
ylabel('Density')
xlabel('B\_hat\_1')
hold off
