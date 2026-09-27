% Exercise - Function - Auxiliary

% This function estimates the auxiliary parameter from a given time 
% series. Specifically, it fits an AR(1) model and extracts the 
% autoregressive coefficient (beta).
function beta_hat = auxiliary(y)
    %% Specify AR(1) model structure
    model = arima(1,0,0);
    %% Estimate AR(1) parameters from input time series
    est = estimate(model,y,'Display','off'); 
    %% Extract estimated AR(1) coefficient (auxiliary parameter)
    beta_hat = est.AR{1}; 
end