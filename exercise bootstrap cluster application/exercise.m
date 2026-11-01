% Exercise - Understanding the method of cluster bootstrap

%% 1. Aim of the exercise
% The aim of this exercise is to understand how to correctly estimate
% standard errors of regression coefficients when observations are
% clustered (e.g., by country), which can lead to biased inference if
% ignored. We compare two methods for addressing this issue: (i) analytical
% cluster-robust estimation using the sandwich formula, and (ii) bootstrap
% resampling of clusters.

%% 2. Theory
% Refer to the accompanying PDF file for the theory.

%% 3. Load data

% 3.1. Clear the memory 
clear;

% 3.2. Import CSV file
data = readtable('data.csv');

%% 4. Obtain OLS statistics for conventional and sandwich estimators

% 4.1. Define response variable
y = table2array(data(:,4));

% 4.2. Create intercept term
x_0 = ones(size(data,1),1);

% 4.3. Create the design matrix
X = [x_0,table2array(data(:,[3,5,6]))];

% 4.4. Obtain OLS statistics 
LSS = exercisefunctionlss(y,X);

%% 5. Estimate standard errors using the conventional estimator

% Estimate standard errors using the conventional estimator
SEE_conventional = LSS.B_hat_SEE;

%% 6. Set parameters for sandwich and bootstrap estimators

% 6.1. Define the number of coefficients
N_k = size(X,2);

% 6.2. Obtain cluster identifiers
cluster_identifiers = data.country;  

% 6.3. Identify unique clusters
unique_clusters = unique(cluster_identifiers);

% 6.4. Count the number of clusters
N_clusters = numel(unique_clusters);

%% 7. Preallocate matrices for sandwich estimator

% 7.1. Initialize cluster-level meat matrix (X_c'*u_c*u_c'*X_c)
M_c = zeros(N_k,N_k);

% 7.2. Initialize sum of meat matrices across clusters
M_c_sum = zeros(N_k,N_k);

% 7.3. Initialize cluster-level design matrix cross-product (X_c'*X_c)
X_cTX_c = zeros(N_k,N_k);

% 7.4. Initialize sum of design matrix cross-products across clusters
X_cTX_c_sum = zeros(N_k,N_k);

%% 8. Create cluster-level design matrix for sandwich estimator

% 8.1. Loop over each cluster to compute meat and bread components
for i = 1:N_clusters
    % Identify current cluster
    cluster_identifier = unique_clusters(i);

    % Find rows corresponding to the current cluster
    cluster_rows = strcmp(data.country,cluster_identifier);

    % Extract residuals for the current cluster
    e_hat_c = LSS.u_hat(cluster_rows);
    
    % Extract design matrix rows for the current cluster
    X_c = X(cluster_rows,:);
    
    % Compute cluster-level meat matrix
    M_c = X_c'*(e_hat_c*e_hat_c')*X_c;
    
    % Accumulate meat matrices
    M_c_sum = M_c_sum+M_c;
    
    % Compute cluster-level design matrix cross-product
    X_cTX_c = X_c'*X_c;
    
    % Accumulate design matrix cross-products
    X_cTX_c_sum = X_cTX_c_sum+X_cTX_c;
end

%% 9. Estimate standard errors using the sandwich estimator

% 9.1. Define the sample size
N_obs = size(data,1);

% 9.2. Apply finite sample correction
df_correction = (N_obs-1)/(N_obs-N_k)*N_clusters/(N_clusters-1);

% 9.3. Compute cluster-robust standard errors using the sandwich formula
SEE_sandwich = df_correction*sqrt( ...
    diag(inv(X_cTX_c_sum)*M_c_sum*inv(X_cTX_c_sum))); 

%% 10. Draw bootstrap samples from the initial sample 

% 10.1. Set the number of bootstrap samples
N_sim = 1000;

% 10.2. Preallocate vector to store coefficient estimates
B_hats_data_samples_boot = NaN(N_k,N_sim);

% 10.3 Sample clusters with replacement and estimate coefficients
for i = 1:N_sim
    % Randomly sample clusters with replacement
    sampled_cluster_identifiers = datasample(unique_clusters, ...
        N_clusters,'Replace',true);

    % Preallocate cell array for selected cluster data
    resampled_data_by_cluster = cell(numel( ...
        sampled_cluster_identifiers),1);

    % Extract rows corresponding to the sampled cluster
    for j = 1:numel(sampled_cluster_identifiers)
        resampled_data_by_cluster{j} = data(strcmp( ...
            data.country,sampled_cluster_identifiers(j)),:);
    end
    
    % Combine data from all selected clusters into one table
    resampled_data = vertcat(resampled_data_by_cluster{:});
    
    % Define response variable
    y = table2array(resampled_data(:,4));
    
    % Create intercept term
    x_0 = ones(size(resampled_data,1),1);
    
    % Create the design matrix
    X = [x_0,table2array(resampled_data(:,[3,5,6]))];
    
    % Compute OLS statisitics
    LSS = exercisefunctionlss(y,X);
    
    % Extract OLS coefficient estimates
    B_hats_data_samples_boot(:,i) = LSS.B_hat;
end

%% 11. Estimate standard errors using bootstrap resampling

% Std. dev. of the bootstrap sampling distribution: Bootstrap estimates
SEE_bootstrap = std(B_hats_data_samples_boot,0,2);
