% Exercise - Understanding the method of paired bootstrap

%% 1. Aim of the exercise
% To understand how to correctly estimate standard errors of regression
% coefficients when the error variance is not constant (i.e.,
% heteroskedasticity is present). Standard inference methods can be
% misleading in such cases. We use the paired bootstrap method, which
% resamples observations as (y, X) pairs, to account for heteroskedasticity
% and compare its performance to sampling directly from the population.

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
X_pop = [random('Uniform',-1,1,[N_obs_pop,1])];

% 4.3. Heteroskedasticity parameter 
Gamma = 1.5; 

% 4.4. Generate error
u_pop = random('Normal',0,exp(X_pop*Gamma),[N_obs_pop,1]);

% 4.5. Set true beta of the model
B_true = 0.5; 

% 4.6. Generate y
y_pop = X_pop*B_true+u_pop;

% 4.7. Create the (paired) population data
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

% 6.3. Preallocate vector to store coeficient estimates
B_hats_data_samples_pop = NaN(N_sim,1);

% 6.4. Monte Carlo sampling
for i = 1:N_sim
    data_samples_pop(:,:,i) = datasample(data_pop,N_obs_sample, ...
        'Replace',false); % Save samples in 3rd dimension
    y = data_samples_pop(:,1,i);
    X = data_samples_pop(:,2,i);
    LSS = exercisefunctionlssrobust(y,X);
    B_hats_data_samples_pop(i) = LSS.B_hat(1,1);
end

%% 7. Pick an "initial" sample

% 7.1. Randomly pick one sample index
sample_index = randi(N_sim);  

% 7.2. Pick one sample from the previously drawn samples
data_sample = data_samples_pop(:,:,sample_index);

%% 8. Draw (bootsrap) samples from the initial sample 

% 8.1. Preallocate vector to store coefficient estimates
B_hats_data_samples_boot = NaN(N_sim,1);

% 8.6. Resample from initial sample and estimate the coefficient 
for i = 1:N_sim
    data_samples_boot = datasample(data_sample,N_obs_sample, ...
        'Replace',true);
    y = data_samples_boot(:,1);
    X = data_samples_boot(:,2);
    LSS = exercisefunctionlssrobust(y,X); 
    B_hats_data_samples_boot(i) = LSS.B_hat(1,1); 
end

%% 9. Plot estimated PDFs
figure
hold on
[f_boot,x_boot] = ksdensity(B_hats_data_samples_boot,'function','pdf');
plot(x_boot,f_boot,'Color',[0.8500,0.3250,0.0980], ...
    'DisplayName','Bootstrap sampling');
xline(mean(B_hats_data_samples_boot),'Color',[0.8500,0.3250,0.0980], ...
    'DisplayName','B\_hat\_boot');
[f_pop,x_pop] = ksdensity(B_hats_data_samples_pop,'function','pdf');
plot(x_pop,f_pop,'Color',[0,0.4470,0.7410], ...
    'DisplayName','Population sampling');
xline(B_true,'Color',[0,0.4470,0.7410], ...
    'DisplayName','B\_true');
ylabel('Density');
xlabel('B\_hat');
legend('show')
title(['Fig. 2. PDF comparison: bootstrap B\_hat vs. ' ...
    'population sample B\_hat']);
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
title(['Fig. 3. CDF comparison: bootstrap B\_hat vs. ' ...
    'population sample B\_hat']);
hold off

%% 11. Comparing the SE estimators across different sample sizes

% 11.1. True estimate based on Monte Carlo simulation
SE_true = std(B_hats_data_samples_pop);

% 11.2. Heteroskedasticity-consistent estimate
SE_HC = LSS.B_hat_SEE_robust;

% 11.3. Bootstrap estimate
SE_boot = std(B_hats_data_samples_boot);
