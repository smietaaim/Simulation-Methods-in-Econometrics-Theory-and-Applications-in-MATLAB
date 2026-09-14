function draws = StochasticRepresentation(N,nu)
%STOCHASTICREPRESENTATION Generate draws from a Student's t distribution.
% Version 1.0, September 2026
% Compatible with MATLAB R2014a and later
%
% Authors: Akash Boelens, Renata-Maria Istrătescu, Tunga Kantarcı
%
% Description:
%   This function generates random draws from a Student's t
%   distribution using its stochastic representation. The t-distributed
%   random variable is constructed from independent standard normal and
%   chi-squared random variables.
%
% Syntax:
%   draws = StochasticRepresentation(N,nu)
%
% Inputs:
%   N  - Number of random draws to generate.
%   nu - Degrees of freedom. Must be positive.
%
% Outputs:
%   draws - Vector containing t-distributed random draws.
%
% Notes:
%   - The stochastic representation of the Student's t-distribution is
%
%                 Z     
%         T = ----------
%             sqrt(U/nu)
%
%     where
%
%         Z ~ N(0,1)
%
%     and
%
%         U ~ Chi-square(nu).
%
%   - The random variables Z and U are assumed to be independent.
%
% ---------- BEGIN FUNCTION BODY BELOW ----------

% Generate standard normal random variables
Z = random('Normal',0,1,N,1);

% Generate chi-squared random variables
U = random('Chi-square',nu,N,1);

% Construct t-distributed random variables
draws = Z ./ sqrt(U./nu);

end