% -------------------------------------------------------------------------
% PURPOSE: Generate Halton draws for quasi-random sampling.
% -------------------------------------------------------------------------

% -------------------------------------------------------------------------
% DEFINITION: N_i — total number of individuals (1×1).
% -------------------------------------------------------------------------
N_i = size(unique(id),1); 

% -------------------------------------------------------------------------
% DEFINITION:
%     halton_N_burns             — Number of initial draws to discard.
%                                  The halton.m function burns 0 by
%                                  default.
%
%     halton_N_dimensions        — Number of prime bases used.
%
%     halton_N_draws_per_i       — Number of Halton draws per individual i.
%
%     halton_N_leaps             — (not used)
%
%     halton_primes              — Prime base (e.g., 2 or 3).
%
%     halton_randomize_indicator — (not used)
%
%     halton_scramble_indicator  — (not used)
% -------------------------------------------------------------------------
halton_N_burns = 50; 
halton_N_dimensions = 1; 
halton_N_draws_per_i = 50;
halton_N_leaps = 0;
halton_primes = 3;
halton_randomize_indicator = 0; 
halton_scramble_indicator = 0;

% -------------------------------------------------------------------------
% DEFINITION: H — Halton quasirandom sequence of size
%                 N_i × halton_N_dimensions × halton_N_draws_per_i.
% -------------------------------------------------------------------------
[~,Z] = halton(N_i,halton_N_dimensions,halton_N_draws_per_i,...
    'prime',halton_primes,...
    'burn',halton_N_burns,...
    'leap',halton_N_leaps,...
    'random',halton_randomize_indicator,...
    'scramble',halton_scramble_indicator);

% -------------------------------------------------------------------------
% Housekeeping: clear Halton parameter variables.
% -------------------------------------------------------------------------
clearvars id N_i halton_N_burns halton_N_dimensions... 
          halton_N_draws_per_i halton_N_leaps halton_primes...
          halton_randomize_indicator halton_scramble_indicator
