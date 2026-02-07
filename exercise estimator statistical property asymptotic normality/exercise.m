% Exercise - Understanding the asymptotic normality of the OLS estimator

%% 1. Aim of the exercise 

% The aim of this exercise is to examine how the sampling distribution of
% the OLS estimator becomes approximately normal as the sample size grows,
% despite the regression errors being non-normal.

%% 2. Clear the workspace

% Clear previous results
clear;

%% 3. Simulation setup

% 3.1. Define a sequence of sample sizes
N_obs_grid = [20 1000];

% 3.2. Define the number of simulations 
N_sim = 5000;

% 3.3. Define the degrees of freedom for t distribution
t_df = 4;

% 3.4. Define true values for the coefficients
B_true = [0.2 3.5]';

%% 4. Plot the error distributions: Normal vs. t

% 4.1. Draw errors from the normal distribution
u_normal = random('Normal',0,1,[100 1]);

% 4.2. Draw errors from the t distribution
u_t = random('t',t_df,[100 1]);

% 4.3. Plot the error distributions
figure
hold on
ksdensity(u_normal)
ksdensity(u_t)
title('Fig. 1. Error distributions: Normal vs. t-distributed errors')
legend('Normal errors','t-distributed errors')
xlabel('u')
ylabel('Density')
hold off

%% 5. Asymptotic normality of the OLS estimator

% 5.1. Preallocate matrix to store scaled estimation errors
B_hat_by_N = NaN(N_sim,length(N_obs_grid));

% 5.2. Loop over each sample size and simulate its sampling distribution
for j = 1:length(N_obs_grid)
    N_obs_j = N_obs_grid(j);
    x_0 = ones(N_obs_j,1);
    x_1 = random('Uniform',-1,1,[N_obs_j 1]);
    X_j = [x_0 x_1];
    B_hat_temp = NaN(N_sim,1);
    for i = 1:N_sim
        u = random('t',t_df,[N_obs_j 1]);
        y = X_j*B_true + u;
        LSS = exercisefunctionlss(y,X_j);
        % Store the asymptotically scaled slope estimation error
        B_hat_temp(i) = sqrt(N_obs_j)*(LSS.B_hat(2,1)-B_true(2));
    end
    % Save results for this sample size
    B_hat_by_N(:,j) = B_hat_temp;
end

%% 6. Overlay convergence to normality

% 6.1. Compute population variance of the t-distributed errors
var_u = t_df/(t_df - 2);

% 6.2. Compute population variance of the uniform regressor
var_x_1 = 1/3;

% 6.3. Compute asymptotic standard deviation of the scaled OLS estimator
std_asymptotic = sqrt(var_u/var_x_1);

% 6.4. Create grid of x-values for plotting the theoretical density
x_grid = linspace(min(B_hat_by_N(:)),max(B_hat_by_N(:)),400);

% 6.5. Construct theoretical asymptotic normal density
pdf_normal = pdf('Normal',x_grid,0,std_asymptotic);

% 6.6. Figure
figure
hold on
plot(x_grid,pdf_normal,'Color',[0 0 0]);
for j = 1:length(N_obs_grid)
    [f,xi] = ksdensity(B_hat_by_N(:,j));
    plot(xi,f)
end
xlim([min(B_hat_by_N(:)) max(B_hat_by_N(:))])
xlabel('B\_hat\_1')
ylabel('Density')
legend(["Asymptotic normal",cellstr("N = " + string(N_obs_grid))])
title(['Fig. 2. Asymptotic normality of the OLS estimator with ' ...
       't-distributed errors'])
hold off
