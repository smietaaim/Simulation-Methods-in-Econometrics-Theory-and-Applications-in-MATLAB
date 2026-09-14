function y = PDF(x,nu)
%PDF Evaluate the PDF of the Student's t distribution.
% Version 1.0, September 2026
% Compatible with MATLAB R2014a and later
%
% Authors: Akash Boelens, Renata-Maria Istrătescu, Tunga Kantarcı
%
% Description:
%   This function evaluates the probability density function (PDF) of
%   the Student's t distribution using its analytical expression. The
%   gamma function is evaluated numerically using Euler's integral.
%
% Syntax:
%   y = PDF(x,nu)
%
% Inputs:
%   x  - Scalar, vector, or matrix of evaluation points.
%   nu - Degrees of freedom. Must be positive.
%
% Outputs:
%   y  - Values of the Student's t-distribution PDF evaluated at x.
%
% Notes:
%   - The Student's t-distribution PDF is given by
%
%                Gamma((nu+1)/2)
%         f(x) = ------------------------- * (1 + x.^2/nu).^(-(nu+1)/2)
%                sqrt(pi*nu) * Gamma(nu/2)
%
%   - The gamma function is computed using Euler's integral
%
%         Gamma(z) = integral_0^Inf t^(z-1) exp(-t) dt.
%
% ---------- BEGIN FUNCTION BODY BELOW ----------

% Compute Gamma((nu + 1)/2)
GammaNuPlusOneOverTwo = ...
    integral(@(t) t.^((nu + 1)/2 - 1) .* exp(-t),0,Inf);

% Compute Gamma(nu/2)
GammaNuOverTwo = ...
    integral(@(t) t.^(nu / 2 - 1) .* exp(-t),0,Inf);

% Evaluate the Student's t-distribution PDF
y = (GammaNuPlusOneOverTwo ./ ...
    (sqrt(nu*pi) .* GammaNuOverTwo)) .* ...
    (1 + x.^2/nu).^(-(nu + 1)/2);

end
