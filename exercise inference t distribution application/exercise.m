% Exercise - Understanding the t distribution using simulation

%% 1. Clear workspace

% Clear
clear;

%% 2. Define parameter values

% 2.1. Define degrees of freedom
nu5 = 5;
nu9 = 9;
nu30 = 30;

% 2.2. Define common x-axis range
xMin = -6;
xMax = 6;

% 2.3. Define histogram bin edges
binEdges = xMin:0.01:xMax;

% 2.4. Define PDF evaluation points
x = xMin:0.01:xMax;

%% 3. Simulating the t distribution from its stochastic representation

% 3.1. Number of random draws
N = 100;

% 3.3. Construct t-distributed random variables
draws = StochasticRepresentation(N,nu5); % In MATLAB, random('T',nu,N,1)

% 3.4. Plot histogram
figure;
set(gcf,'Position',[100 100 1000 1000]);
histogram(draws,...
    'BinEdges',binEdges,...
    'FaceColor',[0.700 0.700 0.700],...
    'EdgeAlpha',0,...
    'DisplayName','All simulated values');
xlim([xMin xMax]);
title(['Fig. 1. Histogram of simulated t-distributed draws', ...
       ', N = ',num2str(N), ...
       ', \nu = ',num2str(nu5)]);
xlabel('t');
ylabel('Frequency');
legend('show');

%% 4. Increasing the number of simulated draws

% 4.1. Number of random draws
N = 1000000;

% 4.2. Construct t-distributed random variables
draws = StochasticRepresentation(N,nu5);

% 4.3. Plot histogram
figure;
set(gcf,'Position',[100 100 1000 1000]);
histogram(draws,...
    'BinEdges',binEdges,...
    'FaceColor',[0.700 0.700 0.700],...
    'EdgeAlpha',0,...
    'DisplayName','All simulated values');
xlim([xMin xMax]);
title(['Fig. 2. Histogram of simulated t-distributed draws', ...
       ', N = ',num2str(N), ...
       ', \nu = ',num2str(nu5)]);
xlabel('t');
ylabel('Frequency');
legend('show');

%% 5. Evaluating the theoretical PDF of the t distribution

% 5.1. Evaluate the PDF of the t distribution
pdfNu5 = PDF(x,nu5); % In MATLAB, pdf('T',x,nu5)

% 5.2. Plot the PDF
figure;
set(gcf,'Position',[100 100 1000 1000]);
plot(x,pdfNu5,...
    'DisplayName','PDF');
xlim([xMin xMax]);
title(['Fig. 3. Theoretical PDF of the t distribution, \nu = ', ...
    num2str(nu5)]);
xlabel('t');
ylabel('Density');
legend('show');

%% 6. Effect of the degrees of freedom on the PDF

% 6.1. Evaluate the PDFs of the t distribution
pdfNu5 = PDF(x,nu5);
pdfNu9 = PDF(x,nu9);
pdfNu30 = PDF(x,nu30);

% 6.2. Plot PDFs
figure;
set(gcf,'Position',[100 100 1000 1000]);
hold on
plot(x,pdfNu5,...
    'DisplayName','\nu = 5');
plot(x,pdfNu9,...
    'DisplayName','\nu = 9');
plot(x,pdfNu30,...
    'DisplayName','\nu = 30');
xlim([xMin xMax]);
title(['Fig. 4. Theoretical PDFs of the t distribution for' ...
    ' different degrees of freedom']);
xlabel('t');
ylabel('Density');
legend('show');

%% 7. Validating the manually computed PDF

% 7.1. Evaluate MATLAB's built-in PDF
yBuiltIn = pdf('T',x,nu5);

% 7.2. Compute the maximum absolute difference
maximumDifference = max(abs(yBuiltIn - pdfNu5));

% 7.3. Check whether the results agree up to numerical precision
PDFsMatch = maximumDifference < 1e-6;
