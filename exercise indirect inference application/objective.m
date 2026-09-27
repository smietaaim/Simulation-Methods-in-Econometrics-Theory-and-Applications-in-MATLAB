% Exercise - Function - Objective

% This function computes the objective value for indirect inference.
% It compares the auxiliary parameter (AR(1) estimate) from observed 
% data with the average auxiliary estimates from simulated MA(1) data.
function Q = objective(theta_candidate,N_sim,T,W,beta_hat)
    %% Preallocate vector to store AR(1) estimates from simulations
    beta_hat_sim = NaN(N_sim,1); 
    %% Simulate MA(1) data and estimate AR(1) parameters
    for i = 1:N_sim
        %% Simulate error terms (white noise)
        epsilon_sim = random("Normal",0,1,[T+1,1]);
        %% Preallocate vector to store simulated MA(1) outcomes
        y_sim = zeros(T,1);
        %% Generate simulated data from MA(1) using theta_candidate
        for t = 2:T+1
            y_sim(t-1) = epsilon_sim(t)+theta_candidate ...
                *epsilon_sim(t-1);
        end
        %% Estimate AR(1) coefficient from simulated data
        beta_hat_sim(i) = auxiliary(y_sim);
    end
    %% Compute average AR(1) estimate across simulations
    beta_tilde = mean(beta_hat_sim);
    %% Calculate the distance between observed and simulated estimates
    Q = (beta_hat-beta_tilde)'*W*(beta_hat-beta_tilde); 
end