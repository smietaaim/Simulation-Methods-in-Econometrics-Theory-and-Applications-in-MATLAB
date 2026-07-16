% -------------------------------------------------------------------------
% PURPOSE: Compute the sum of log simulated likelihoods over the sample.
% -------------------------------------------------------------------------
function sum_log_avg_lik = likelihood(...
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
    X_i,t,Z)

% -------------------------------------------------------------------------
% DEFINITION: N_k — number of covariates (1×1 scalar).
% -------------------------------------------------------------------------
N_k = size(X_i,2);

% -------------------------------------------------------------------------
% DEFINITION:
%     theta      — parameter vector to be estimated.
%
%     beta_l     — coefficient vector on X_i (N_k × 1).
%
%     eta_l      — coefficient on t (1 × 1).
%
%     alpha_y    — preference parameter (1 × 1).
%
%     alpha_l_y  — preference parameter (1 × 1).
%
%     rho        — discount factor (1 × 1).
%
%     sigma_l    — standard deviation of the error term entering the
%                  random‑coefficient equation for hours (1 × 1).
%
%     T          — number of hours in a week; estimated (1 × 1).
% -------------------------------------------------------------------------
beta_l    = theta(1:N_k)';
eta_l     = theta(N_k + 1);
sigma_l   = theta(N_k + 2);
T         = theta(N_k + 3);
alpha_y   = theta(N_k + 4);
alpha_l_y = theta(N_k + 5);
rho       = theta(N_k + 6);

% -------------------------------------------------------------------------
% DEFINITION: e_i_l — error term entering the random‑coefficient equation.
%                     Dimensions: N_i × 1 × halton_N_draws_per_i.
%                     Multiply the standard deviation of the errors by
%                     standard normal draws to obtain normal errors.
% -------------------------------------------------------------------------
e_i_l(:,1,:) = sigma_l * Z(:,1,:);

% -------------------------------------------------------------------------
% DEFINITION: alpha_i_t_l — random preference parameter.
%                           Dimensions: N_i × N_t × N_halton_draws_per_i.
%                           The third dimension reflects multiple draws for
%                           each individual i.
% -------------------------------------------------------------------------
alpha_i_t_l = X_i * beta_l + eta_l * t + e_i_l; 

% -------------------------------------------------------------------------
% DEFINITION: log_T_l_i_t_61E — log(T - l_i_t_61E), N_i × N_t.
% -------------------------------------------------------------------------
log_T_h_i_t_61E = log(T - h_i_t_61E);
log_T_h_i_t_61P = log(T - h_i_t_61P);
log_T_h_i_t_61L = log(T - h_i_t_61L);

log_T_h_i_t_63E = log(T - h_i_t_63E);
log_T_h_i_t_63P = log(T - h_i_t_63P);
log_T_h_i_t_63L = log(T - h_i_t_63L);

log_T_h_i_t_65E = log(T - h_i_t_65E);
log_T_h_i_t_65P = log(T - h_i_t_65P);
log_T_h_i_t_65L = log(T - h_i_t_65L);

% -------------------------------------------------------------------------
% DEFINITION: log_y_i_t_61E — log(y_i_t_61E), dimensions N_i × N_t.
% -------------------------------------------------------------------------
log_y_i_t_61E = log(y_i_t_61E);  
log_y_i_t_61P = log(y_i_t_61P);  
log_y_i_t_61L = log(y_i_t_61L);  
                                    
log_y_i_t_63E = log(y_i_t_63E);  
log_y_i_t_63P = log(y_i_t_63P);  
log_y_i_t_63L = log(y_i_t_63L);  
                                    
log_y_i_t_65E = log(y_i_t_65E);  
log_y_i_t_65P = log(y_i_t_65P);  
log_y_i_t_65L = log(y_i_t_65L);  

% -------------------------------------------------------------------------
% DEFINITION: U_i_t_61E — representative utility, dimensions
%                         N_i × N_t × N_halton_draws_per_i.
%                         The third dimension reflects multiple draws
%                         for each individual i.
% -------------------------------------------------------------------------
U_i_t_61E = alpha_i_t_l .* log_T_h_i_t_61E ...
            + alpha_y .* log_y_i_t_61E ...
            + alpha_l_y .* log_T_h_i_t_61E .* log_y_i_t_61E;

U_i_t_61P = alpha_i_t_l .* log_T_h_i_t_61P ...
            + alpha_y .* log_y_i_t_61P ...
            + alpha_l_y .* log_T_h_i_t_61P .* log_y_i_t_61P;

U_i_t_61L = alpha_i_t_l .* log_T_h_i_t_61L ...
            + alpha_y .* log_y_i_t_61L ...
            + alpha_l_y .* log_T_h_i_t_61L .* log_y_i_t_61L;

U_i_t_63E = alpha_i_t_l .* log_T_h_i_t_63E ...
            + alpha_y .* log_y_i_t_63E ...
            + alpha_l_y .* log_T_h_i_t_63E .* log_y_i_t_63E;

U_i_t_63P = alpha_i_t_l .* log_T_h_i_t_63P ...
            + alpha_y .* log_y_i_t_63P ...
            + alpha_l_y .* log_T_h_i_t_63P .* log_y_i_t_63P;

U_i_t_63L = alpha_i_t_l .* log_T_h_i_t_63L ...
            + alpha_y .* log_y_i_t_63L ...
            + alpha_l_y .* log_T_h_i_t_63L .* log_y_i_t_63L;

U_i_t_65E = alpha_i_t_l .* log_T_h_i_t_65E ...
            + alpha_y .* log_y_i_t_65E ...
            + alpha_l_y .* log_T_h_i_t_65E .* log_y_i_t_65E;

U_i_t_65P = alpha_i_t_l .* log_T_h_i_t_65P ...
            + alpha_y .* log_y_i_t_65P ...
            + alpha_l_y .* log_T_h_i_t_65P .* log_y_i_t_65P;

U_i_t_65L = alpha_i_t_l .* log_T_h_i_t_65L ...
            + alpha_y .* log_y_i_t_65L ...
            + alpha_l_y .* log_T_h_i_t_65L .* log_y_i_t_65L;

% -------------------------------------------------------------------------
% DEFINITION: rho_0_to_40 — vector with elements rho^(t − 60); 0 = age 60,
%                           40 = age 100; dimensions 1 × N_t.
% -------------------------------------------------------------------------
rho_0_to_40 = rho .^ (t - 60);

% -------------------------------------------------------------------------
% DEFINITION: pi — vector of survival probabilities at age 60 for ages
%                  60 to 100; dimensions 1 × N_t.
% -------------------------------------------------------------------------
pi = [1.0000 0.9943 0.9878 0.9809 0.9733 0.9650 0.9564 0.9468 0.9367 ...
      0.9258 0.9138 0.9008 0.8865 0.8705 0.8536 0.8352 0.8152 0.7933 ...
      0.7698 0.7449 0.7173 0.6864 0.6529 0.6173 0.5791 0.5385 0.4943 ...
      0.4491 0.4027 0.3545 0.3076 0.2612 0.2188 0.1779 0.1406 0.1071 ...
      0.0808 0.0586 0.0405 0.0277 0.0186];

% -------------------------------------------------------------------------
% DEFINITION: U_i_61E — discounted representative utility, dimensions
%                       N_i × 1 × halton_N_draws_per_i.
% -------------------------------------------------------------------------
U_i_61E = sum(rho_0_to_40 .* pi .* U_i_t_61E,2);
U_i_61P = sum(rho_0_to_40 .* pi .* U_i_t_61P,2);
U_i_61L = sum(rho_0_to_40 .* pi .* U_i_t_61L,2);
                                               
U_i_63E = sum(rho_0_to_40 .* pi .* U_i_t_63E,2);
U_i_63P = sum(rho_0_to_40 .* pi .* U_i_t_63P,2);
U_i_63L = sum(rho_0_to_40 .* pi .* U_i_t_63L,2);
                                                 
U_i_65E = sum(rho_0_to_40 .* pi .* U_i_t_65E,2);
U_i_65P = sum(rho_0_to_40 .* pi .* U_i_t_65P,2);
U_i_65L = sum(rho_0_to_40 .* pi .* U_i_t_65L,2);

% -------------------------------------------------------------------------
% DEFINITION: exp_U_i_61E — component of the choice probability, dimensions
%                           N_i × 1 × N_halton_draws_per_i.
% -------------------------------------------------------------------------
exp_U_i_61E = exp(U_i_61E);   
exp_U_i_61P = exp(U_i_61P);   
exp_U_i_61L = exp(U_i_61L);   
                                 
exp_U_i_63E = exp(U_i_63E);   
exp_U_i_63P = exp(U_i_63P);   
exp_U_i_63L = exp(U_i_63L);   
                                 
exp_U_i_65E = exp(U_i_65E);   
exp_U_i_65P = exp(U_i_65P);   
exp_U_i_65L = exp(U_i_65L);   

% -------------------------------------------------------------------------
% DEFINITION: exp_U_i_61 — component of the choice probability, 
%                          N_i × N_j_plans × N_halton_draws_per_i.
% -------------------------------------------------------------------------
exp_U_i_61 = [exp_U_i_61E exp_U_i_61P exp_U_i_61L];
exp_U_i_63 = [exp_U_i_63E exp_U_i_63P exp_U_i_63L];
exp_U_i_65 = [exp_U_i_65E exp_U_i_65P exp_U_i_65L];

% -------------------------------------------------------------------------
% DEFINITION: N_j — number of choice alternatives.
% -------------------------------------------------------------------------
N_j = 3;

% -------------------------------------------------------------------------
% DEFINITION: ind_choice_at_age_regime_61 — column vectors; each column is
%                                           a dummy indicating if
%                                           alternative j is chosen by all
%                                           i; each row indicates which
%                                           alternative is chosen by i; N_i
%                                           × N_j_plans.
% -------------------------------------------------------------------------
ind_choice_at_age_regime_61(:,N_j - 2) = (choice_at_age_regime_61 == 1); 
ind_choice_at_age_regime_61(:,N_j - 1) = (choice_at_age_regime_61 == 2); 
ind_choice_at_age_regime_61(:,N_j - 0) = (choice_at_age_regime_61 == 3); 

ind_choice_at_age_regime_63(:,N_j - 2) = (choice_at_age_regime_63 == 1); 
ind_choice_at_age_regime_63(:,N_j - 1) = (choice_at_age_regime_63 == 2); 
ind_choice_at_age_regime_63(:,N_j - 0) = (choice_at_age_regime_63 == 3); 

ind_choice_at_age_regime_65(:,N_j - 2) = (choice_at_age_regime_65 == 1); 
ind_choice_at_age_regime_65(:,N_j - 1) = (choice_at_age_regime_65 == 2); 
ind_choice_at_age_regime_65(:,N_j - 0) = (choice_at_age_regime_65 == 3); 

% -------------------------------------------------------------------------
% DEFINITION: lik_choice_at_age_regime_61 — likelihood of choice at
%                                           retirement-age regime 61, for
%                                           each i; N_i × 1 ×
%                                           N_halton_draws_per_i. The sum
%                                           function takes an array of size
%                                           N_i × N_j_plans ×
%                                           N_halton_draws_per_i and sums
%                                           across dimension 2, returning
%                                           an N_i × 1 ×
%                                           N_halton_draws_per_i column
%                                           vector.
% -------------------------------------------------------------------------
lik_choice_at_age_regime_61 = ...
    sum(exp_U_i_61 .* ind_choice_at_age_regime_61,2) ./ ...
    sum(exp_U_i_61,2);

lik_choice_at_age_regime_63 = ...
    sum(exp_U_i_63 .* ind_choice_at_age_regime_63,2) ./ ...
    sum(exp_U_i_63,2);

lik_choice_at_age_regime_65 = ...
    sum(exp_U_i_65 .* ind_choice_at_age_regime_65,2) ./ ...
    sum(exp_U_i_65,2);

% -------------------------------------------------------------------------
% DEFINITION: lik — joint probability of choosing plans in multiple
%                   questions for each i; N_i × 1 × N_halton_draws_per_i.
% -------------------------------------------------------------------------
lik = lik_choice_at_age_regime_61 .* ...
      lik_choice_at_age_regime_63 .* ...
      lik_choice_at_age_regime_65; 

% -------------------------------------------------------------------------
% DEFINITION: avg_lik — simulated likelihood for each i; N_i × 1. The mean
%                       is taken over the third dimension of "lik", which
%                       corresponds to multiple draws for each i.
% -------------------------------------------------------------------------
avg_lik = mean(lik,3); 

% -------------------------------------------------------------------------
% DEFINITION: log_avg_lik — log of the simulated likelihood for each i; 
%                           N_i × 1.
% -------------------------------------------------------------------------
log_avg_lik = log(avg_lik);

% -------------------------------------------------------------------------
% DEFINITION: sum_log_avg_lik — sum of the log simulated likelihoods in the
%                               sample; 1 × 1.
% -------------------------------------------------------------------------
sum_log_avg_lik = -sum(log_avg_lik);

end
