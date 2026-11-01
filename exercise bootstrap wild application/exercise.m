% Exercise - Understanding the method of wild bootstrap

%% 1. Aim of the exercise
% The aim of this exercise is to understand how to accurately estimate the
% standard errors of regression coefficients in the presence of
% heteroskedastic error variance. Traditional bootstrap methods, such as
% the paired bootstrap, may yield misleading inference under
% heteroskedasticity. To address this, we use the wild bootstrap method,
% which improves robustness by modifying residuals with random multipliers.
% We then compare its performance to that of direct sampling from the
% population.

%% 2. Theory
% Refer to the accompanying PDF file for the theory.

%% 3. Set values for the parameters of the simulation

% 3.1. Clear the memory
clear;

% 3.2. Set the number of simulations as the number of bootstrap samples
N_sim = 5000;

%% 4. Generate population data

% 4.1. Set the population size
N_obs_pop = 10000;

% 4.2. Generate data for the independent variable
X_pop = random('Uniform',-1,1,[N_obs_pop,1]);

% 4.3. Heteroskedasticity parameter 
Gamma = 1.5; 

% 4.4. Generate error
u_pop = random('Normal',0,exp(X_pop*Gamma),[N_obs_pop,1]);

% 4.5. Set true beta of the model
B_true = 0.5; 

% 4.6. Generate y
y_pop = X_pop*B_true+u_pop;

% 4.7. Generate (paired) population data
data_pop = [y_pop X_pop];

%% 5. Plot the heteroskedastic true data
figure
hold on
scatter(X_pop,y_pop);
title('Fig. 1. Simulated heteroskedastic data');
xlabel('X');
ylabel('y');
legend('Population data');
hold off

%% 6. Draw samples from the population

% 6.1. Set the sample size
N_obs_sample = 100;

% 6.2. Preallocate matrix to save (paired) samples
data_samples_pop = NaN(N_obs_sample,2,N_sim);

% 6.3. Preallocate vector to store coeficient estimatess
B_hats_data_samples_pop = NaN(N_sim,1);

% 6.4. Monte Carlo sampling
for i = 1:N_sim
    data_samples_pop(:,:,i) = datasample(data_pop,N_obs_sample, ...
        'Replace',false); % Save samples in 3rd dimension
    y = data_samples_pop(:,1,i);
    X = data_samples_pop(:,2,i);
    LSS = exercisefunctionlss(y,X);
    B_hats_data_samples_pop(i) = LSS.B_hat(1,1);
end

%% 7. Pick an "initial" sample

% 7.1. Randomly pick one sample index
sample_index = randi(N_sim);  

% 7.2. Pick one sample from the previously drawn samples
data_sample = data_samples_pop(:,:,sample_index);

%% 8. Draw (bootstrap) samples from the initial sample 

% 8.1. Extract fixed design matrix and outcome variable
y = data_sample(:,1);
X = data_sample(:,2);

% 8.2. Estimate beta
LSS = exercisefunctionlss(y,X); 
B_hat = LSS.B_hat(1,1);

% 8.3. Compute the residuals
u_hat = LSS.u_hat;

% 8.4. Compute leverage values used in HC2 correction
h = diag(X/(X'*X)*X');

% 8.5. Preallocate vector to store coefficient estimates
B_hats_data_samples_boot = NaN(N_sim,1);

% 8.6. Resample from initial sample and estimate the coefficient 
for i = 1:N_sim
    v = random('Normal',0,1,[N_obs_sample,1]); % N(0,1) multipliers
    % v = 2*(random('Uniform',-1,1,[N_obs_sample,1]) > 0.5)-1; % Rademacher multipliers
    y_boot = X*B_hat+(u_hat./sqrt(1-h)).*v; % u_hat./sqrt(1-h) is the HC2 adjustment to residuals
    LSS = exercisefunctionlss(y_boot,X); 
    B_hats_data_samples_boot(i) = LSS.B_hat(1,1);
end

%% 9. Plot estimated PDFs
figure
hold on
[f_boot,x_boot] = ksdensity(B_hats_data_samples_boot,'function','pdf');
plot(x_boot,f_boot,'Color',[0.8500,0.3250,0.0980], ...
    'DisplayName','Bootstrap sampling');
xline(mean(B_hats_data_samples_boot),'Color',[0.8500,0.3250,0.0980], ...
    'DisplayName','B\_hat\_bootstrap');
[f_pop,x_pop] = ksdensity(B_hats_data_samples_pop,'function','pdf');
plot(x_pop,f_pop,'Color',[0,0.4470,0.7410], ...
    'DisplayName','Population sampling');
xline(B_true,'Color',[0,0.4470,0.7410], ...
    'DisplayName','B\_true');
ylabel('Density');
xlabel('B\_hat');
legend('show')
title(['Fig. 2. PDF comparison: bootstrap vs. ' ...
    'population sampling']);
hold off

%% 10. Plot estimated CDFs
figure
hold on
[f_boot,x_boot] = ksdensity(B_hats_data_samples_boot,'function','cdf');
plot(x_boot,f_boot,'Color',[0,0.4470,0.7410], ...
    'DisplayName','Population sampling');
[f_pop,x_pop] = ksdensity(B_hats_data_samples_pop,'function','cdf');
plot(x_pop,f_pop,'Color',[0.8500,0.3250,0.0980], ...
    'DisplayName','Bootstrap sampling');
ylabel('Cumulative distribution');
xlabel('B\_hat');
legend('show');
title(['Fig. 3. CDF comparison: bootstrap vs. ' ...
    'population sampling']);
hold off
