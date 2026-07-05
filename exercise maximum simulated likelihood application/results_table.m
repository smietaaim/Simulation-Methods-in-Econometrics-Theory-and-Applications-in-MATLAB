% -------------------------------------------------------------------------
% PURPOSE: Summarize and display parameter estimates in a formatted table.
% -------------------------------------------------------------------------

% -------------------------------------------------------------------------
% Format style
% -------------------------------------------------------------------------
format bank

% -------------------------------------------------------------------------
% DEFINITION: Parameter — names of estimated parameters.
% -------------------------------------------------------------------------
Parameter = [...
    "beta_l: Constant";...
    "beta_l: Male";...
    "beta_l: Home owner";...
    "beta_l: Household with no children";...
    "beta_l: With partner";...
    "beta_l: Age";...
    "beta_l: High education";...
    "beta_l: Attachment";...
    "beta_l: Early retirement";...
    "beta_l: Health short term";...
    "eta_l";...
    "alpha_y";...
    "alpha_l_y";...
    "rho";...
    "sigma_l";...
    "T"];

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
