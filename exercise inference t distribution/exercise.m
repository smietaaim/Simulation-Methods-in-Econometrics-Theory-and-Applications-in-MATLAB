% Exercise - Understanding the t distribution using simulation

%% 1. Clear workspace

% Clear
clear;

%% 2. Define parameter values

% Define degrees of freedom
nu5 = 5;

%% 3. Simulating the t distribution from its stochastic representation

% 3.1. Number of random samples
N = 1000;

% 3.2. Generate standard normal and chi-squared random variables
Z = random('Normal',0,1,N,1);
U = random('Chi-square',nu5,N,1);

% 3.3. Construct t-distributed random variables
samples = Z ./ sqrt(U/nu5); % Alternatively, samples = random('T',nu,N,1);

% 3.4. Plot histogram
figure;
set(gcf,'Position',[100 100 1000 1000]);
histogram(samples, ...
    'BinLimits',[-6,6], ...
    'NumBins',30, ...
    'DisplayName','All simulated values');
title(['Fig. 1. Histogram of simulated t-distributed draws, N = ', ...
       num2str(N), ', \nu = ', num2str(nu5)]);
xlabel('t');
ylabel('Frequency');
xlim([-6 6]);
legend('show');

%% 4. Increasing the number of simulated draws

% 4.1. Number of random samples
N = 1000000;

% 4.2. Generate standard normal and chi-squared random variables
Z = random('Normal',0,1,N,1);
U = random('Chi-square',nu5,N,1);

% 4.3. Construct t-distributed random variables
samples = Z ./ sqrt(U / nu5); 

% 4.3. Plot histogram
figure;
set(gcf,'Position',[100 100 1000 1000]);
histogram(samples, ...
    'BinLimits',[-6,6], ...
    'DisplayName','All simulated values');
title(['Fig. 2. Histogram of simulated t-distributed draws, N = ', ...
       num2str(N), ', \nu = ', num2str(nu5)]);
xlabel('t');
ylabel('Frequency');
xlim([-6 6]);
legend('show');

%% 5. Evaluating the theoretical PDF of the t distribution

% 5.1. Define evaluation points
x = -6:0.1:6;

% 5.2. Evaluate the PDF of the t distribution
y = PDF(x,nu5);

% 5.3. Plot the PDF
figure;
set(gcf,'Position',[100 100 1000 1000]);
plot(x,y,...
    '-k',...
    'DisplayName','PDF');
title(['Fig. 3. Theoretical PDF of the t distribution, \nu = ', ...
    num2str(nu5)]);
xlabel('t');
ylabel('Density');
xlim([-6 6]);
legend('show');

%% 6. Effect of the degrees of freedom on the PDF

% 6.1. Define evaluation points
x = -6:0.1:6;

% 6.2. Define degrees of freedom
nu9 = 9;
nu30 = 30;

% 6.3. Evaluate the PDFs
pdfNu5 = PDF(x,nu5);
pdfNu9 = PDF(x,nu9);
pdfNu30 = PDF(x,nu30);

% 6.4. Plot PDFs
figure;
set(gcf,'Position',[100 100 1000 1000]);
hold on
plot(x,pdfNu5,...
    'DisplayName','\nu = 5');
plot(x,pdfNu9,...
    'DisplayName','\nu = 9');
plot(x,pdfNu30,...
    'DisplayName','\nu = 30');
title(['Fig. 4. Theoretical PDFs of the t distribution for' ...
    ' different degrees of freedom']);
xlabel('t');
ylabel('Density');
xlim([-6 6]);
legend('show');

%% 7. Validating the manually computed PDF

% 7.1. Evaluate MATLAB's built-in PDF
yBuiltIn = pdf('T',x,nu5);

% 7.2. Compute the maximum absolute difference
maximumDifference = max(abs(yBuiltIn - y));

% 7.3. Check whether the results agree up to numerical precision
PDFsMatch = maximumDifference < 1e-6;
