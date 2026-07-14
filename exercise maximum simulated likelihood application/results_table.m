% -------------------------------------------------------------------------
% PURPOSE: Summarize and display parameter estimates in a formatted table.
% -------------------------------------------------------------------------

% -------------------------------------------------------------------------
% Format style
% -------------------------------------------------------------------------
format short

% -------------------------------------------------------------------------
% DEFINITION: Parameter — names of estimated parameters.
% -------------------------------------------------------------------------
Parameter = [...
    "beta_l: constant";...
    "beta_l: age";...
    "beta_l: male";...
    "beta_l: high education";...
    "beta_l: household with no children";...
    "beta_l: with partner";...
    "beta_l: home owner";...
    "beta_l: health short term";...
    "beta_l: attachment";...
    "beta_l: early retirement";...
    "eta_l";...
    "sigma_l";...
    "T";...
    "alpha_y";...
    "alpha_l_y";...
    "rho"];

% -------------------------------------------------------------------------
% DEFINITION: Coefficient — point estimates of parameters.
% -------------------------------------------------------------------------
Coefficient = theta_hat';

% -------------------------------------------------------------------------
% DEFINITION: SE — standard errors of parameter estimates.
% -------------------------------------------------------------------------
SE = theta_hat_see';

% -------------------------------------------------------------------------
% DEFINITION: t_statistic — t-statistics for parameter estimates.
% -------------------------------------------------------------------------
t_statistic = t_value';

% -------------------------------------------------------------------------
% DEFINITION: CI — 95% confidence intervals.
% -------------------------------------------------------------------------
CI;

% -------------------------------------------------------------------------
% Table of parameter estimates.
% -------------------------------------------------------------------------
table(Parameter,Coefficient,SE,t_statistic,CI)
