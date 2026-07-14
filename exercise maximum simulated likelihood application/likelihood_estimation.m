% -------------------------------------------------------------------------
% PURPOSE: Conduct maximum likelihood estimation.
% -------------------------------------------------------------------------

% -------------------------------------------------------------------------
% DEFINITION: obj — objective function for fmincon.
% -------------------------------------------------------------------------
obj = @(theta)likelihood(...
    theta,...
    choice_at_age_regime_61,...
    choice_at_age_regime_63,...
    choice_at_age_regime_65,...
    h_i_t_61E,h_i_t_61P,h_i_t_61L,...
    h_i_t_63E,h_i_t_63P,h_i_t_63L,...
    h_i_t_65E,h_i_t_65P,h_i_t_65L,...
    y_i_t_61E,y_i_t_61P,y_i_t_61L,...
    y_i_t_63E,y_i_t_63P,y_i_t_63L,...
    y_i_t_65E,y_i_t_65P,y_i_t_65L,...
    X_i,t,Z);

% -------------------------------------------------------------------------
% DEFINITION: theta_ig — initial guess for parameter vector theta_hat. The 
%                        parameter order follows the elements of the 
%                        parameter vector theta defined in likelihood.m.
% -------------------------------------------------------------------------
theta_ig = [-7.01 -0.19 -0.75 -0.04 0.03 0.06 0.04 0.05 -0.08 0.19 ...
             0.10 0.42 50.19 -0.57 0.23 1.02];

% -------------------------------------------------------------------------
% DEFINITION: NumberOfParameters — number of parameters in theta_ig.
% -------------------------------------------------------------------------
[~,NumberOfParameters] = size(theta_ig);

% -------------------------------------------------------------------------
% DEFINITION: options — options for fmincon.
% -------------------------------------------------------------------------
options = optimoptions(...
    'fmincon',...
    'Algorithm','sqp',...
    'Diagnostics','off',...
    'Display','iter-detailed',...
    'MaxIterations',400,...
    'MaxFunctionEvaluations',100*NumberOfParameters,...
    'OptimalityTolerance',1e-4,...
    'StepTolerance',1e-4,...
    'SpecifyObjectiveGradient',false,...
    'UseParallel',false,...
    'FunValCheck','off');

% -------------------------------------------------------------------------
% Start stopwatch timer.
% -------------------------------------------------------------------------
tic

% -------------------------------------------------------------------------
% DEFINITION: theta_hat — maximum likelihood estimates of the parameters.
% -------------------------------------------------------------------------
[theta_hat,fval,exitflag,output,~,grad,hessian] = ...
    fmincon(obj,theta_ig,[],[],[],[],[],[],[],options);

% -------------------------------------------------------------------------
% Read elapsed time from stopwatch.
% -------------------------------------------------------------------------
disp(['Elapsed time is ' num2str(round(toc/60,2)) ' minutes.'])

% -------------------------------------------------------------------------
% DEFINITION: theta_hat_see — standard errors of theta_hat. The Hessian
%                             returned by fmincon (for sqp) is an
%                             approximation to the observed negative
%                             Hessian of the log-likelihood; its inverse
%                             gives the variance–covariance matrix. Because
%                             we minimize the negative log-likelihood, the
%                             returned Hessian corresponds to the observed
%                             negative Hessian. See "Hessian Output" in the
%                             fmincon documentation.
% -------------------------------------------------------------------------
theta_hat_see = sqrt(diag(inv(hessian)))';

% -------------------------------------------------------------------------
% DEFINITION: t_value — t-statistics for theta_hat.
% -------------------------------------------------------------------------
t_value = theta_hat ./ theta_hat_see;

% -------------------------------------------------------------------------
% DEFINITION: CI — 95% confidence intervals.
% -------------------------------------------------------------------------
CI = [theta_hat' - 1.96*theta_hat_see', ...
      theta_hat' + 1.96*theta_hat_see'];
