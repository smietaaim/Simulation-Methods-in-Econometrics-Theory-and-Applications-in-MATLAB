% Exercise - Understanding measurement error using simulation

%% 1. Aim of the exercise

% To learn how measurement error leads to a biased OLS estimate.

%% 2. Theory

% Refer to the accompanying PDF file for the theory.

%% 3. Set the parameters of the simulation

% 3.1. Clear the memory 
clear;

% 3.2. Set the number of simulations 
N_sim = 1000;

% 3.3. Set the sample size
N_obs = 1000;

% 3.4. Set true values for the slope
B_true = 0.5; 

% 3.5. Create the systematic component of the regression
X = random("Uniform",-1,1,[N_obs 1]);

% 3.6. Level of measurement error in terms of the SD of random noise
measurement_error_level = 0:0.1:1;

%% 4. Nested for loops for simulation and measurement error level

% 4.1. Preallocate matrix to store OLS coefficient estimates
B_hat = NaN(N_sim,1);

% 4.2. Preallocate matrix to store estimates across mea. err. levels
B_hat_measurement_error_level = NaN(N_sim, ...
    1,length(measurement_error_level));

% 4.3. Nested for loops for simulation and measurement error level
for j = 1:length(measurement_error_level)
    X_with_measurement_error = X+random('Normal',0,1,[N_obs 1]) ...
        *sqrt(measurement_error_level(j)); 
    for i = 1:N_sim
        u = random('Normal',0,1,[N_obs 1]);
        y = X*B_true+u;
        LSS = lss(y,X_with_measurement_error); 
        B_hat(i,1) = LSS.B_hat(1,1);
    end
    B_hat_measurement_error_level(:,:,j) = B_hat;
end

%% 5. Plot the sampling distribution of the OLS estimator

% 5.1. Means of the sampling distributions
mean1 = mean(B_hat_measurement_error_level(:,1,1));
mean2 = mean(B_hat_measurement_error_level(:,1,6));
mean3 = mean(B_hat_measurement_error_level(:,1,11));

% 5.2. Kernel density estimation
[f1,x1] = ksdensity(B_hat_measurement_error_level(:,1,1));
[f2,x2] = ksdensity(B_hat_measurement_error_level(:,1,6));
[f3,x3] = ksdensity(B_hat_measurement_error_level(:,1,11));

% 5.3. Sampling distribution of OLS estimator
figure('Position',[100 100 1000 1000]);
hold on
% Density plots
plot(x1,f1, ...
    'Color',[0.000 0.447 0.741], ...
    'DisplayName','Measurement error variance = 0');
plot(x2,f2, ...
    'Color',[0.850 0.325 0.098], ...
    'DisplayName','Measurement error variance = 0.5');
plot(x3,f3, ...
    'Color',[0.929 0.694 0.125], ...
    'DisplayName','Measurement error variance = 1');
% Means of the distributions
xline(mean1, ...
    'Color',[0.000 0.447 0.741], ...
    'LineStyle',':', ...
    'DisplayName',['Mean (\sigma_v^2 = 0) = ' num2str(mean1,'%.3f')]);
xline(mean2, ...
    'Color',[0.850 0.325 0.098],...
    'LineStyle',':', ...
    'DisplayName',['Mean (\sigma_v^2 = 0.5) = ' num2str(mean2,'%.3f')]);
xline(mean3, ...
    'Color',[0.929 0.694 0.125],...
    'LineStyle',':', ...
    'DisplayName',['Mean (\sigma_v^2 = 1) = ' num2str(mean3,'%.3f')]);
% True value
xline(B_true, ...
    'Color',[0.000 0.000 0.000], ...
    'DisplayName',['True \beta = ' num2str(B_true)]);
xlabel('B\_hat');
ylabel('Density');
title(['Fig. 1. Effect of measurement error on ', ...
       'the sampling distribution of the OLS estimator']);
legend('show')
hold off
