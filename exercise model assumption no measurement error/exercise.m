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
        LSS = exercisefunctionlss(y,X_with_measurement_error); 
        B_hat(i,1) = LSS.B_hat(1,1);
    end
    B_hat_measurement_error_level(:,:,j) = B_hat;
end

%% 5. Plot the sampling distribution of the OLS estimator

% 5.1. Kernel density estimation
[f1,x1] = ksdensity(B_hat_measurement_error_level(:,1,1));
[f2,x2] = ksdensity(B_hat_measurement_error_level(:,1,6));
[f3,x3] = ksdensity(B_hat_measurement_error_level(:,1,11));

% 5.2. Sampling distribution of OLS estimator subject to measurement error
figure
hold on
plot(x1,f1,'DisplayName','Measurement error variance = 0');
plot(x2,f2,'DisplayName','Measurement error variance = 0.5');
plot(x3,f3,'DisplayName','Measurement error variance = 1');
xlabel('B\_hat');
ylabel('Density');
title(['Fig. 1. Effect of measurement error on ' ...
    'the sampling distribution of OLS estimator']);
legend('show');
hold off
