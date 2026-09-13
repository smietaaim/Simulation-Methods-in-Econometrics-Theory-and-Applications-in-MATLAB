% Exercise - Understanding hypothesis tesing using simulation

%% 1. Clear workspace

% Clear
clear;

%% 2. Define t value and degrees of freedom

% Define t value
tValue = 1.8;

% Define degrees of freedom
nu = 20;

%% 2. Generate Student's t distributed values

% 2.1. Number of random samples
N = 20000;

% 2.3. Define histogram bin edges
binEdges = -6:0.01:6;

% 2.4. Generate standard normal and chi-squared random variables
Z = random('Normal',0,1,N,1);
U = random('Chi-square',nu,N,1);

% 2.5. Construct t-distributed random variables
samples = Z ./ sqrt(U / nu); % Alternatively, samples = random('T',nu,N,1);

% 2.6. Create histogram
figure;
set(gcf,'Position',[100 100 1000 1000]);
histogram(samples,...
    'BinEdges',binEdges,...
    'FaceColor','#b3b3b3',...
    'EdgeAlpha',0,...
    'DisplayName','All simulated values');
title(['Fig. 1. Histogram of simulated t-distributed draws, N = ', ...
       num2str(N), ', \nu = ', num2str(nu)]);
xlabel('t');
ylabel('Frequency');
legend('show');

%% 3. Add the test statistic to the plot

% 3.2. Plot histogram and test statistic
figure;
set(gcf,'Position',[100 100 1000 1000]);
histogram(samples,...
    'BinEdges',binEdges,...
    'FaceColor','#b3b3b3',...
    'EdgeAlpha',0,...
    'DisplayName','All simulated values');
hold on
xline(tValue,...
    '-b',...
    'DisplayName','Test statistic');
hold off
title(['Fig. 2. Histogram of simulated t-distributed draws, N = ', ...
       num2str(N),', \nu = ',num2str(nu)]);
xlabel('t');
ylabel('Frequency');
legend('show');

%% 4. Highlight values more extreme than the test statistic

% 4.1. Select values exceeding the test statistic
extremeValues = samples(samples > tValue);

% 4.2. Plot histogram and highlight values exceeding the test statistic
figure;
set(gcf,'Position',[100 100 1000 1000]);
histogram(samples,...
    'BinEdges',binEdges,...
    'FaceColor','#b3b3b3',...
    'EdgeAlpha',0,...
    'DisplayName','All simulated values');
hold on
histogram(extremeValues,...
    'BinEdges',binEdges,...
    'FaceColor','blue',...
    'EdgeAlpha',0,...
    'DisplayName','Values exceeding the test statistic');
xline(tValue,...
    '-b',...
    'DisplayName','Test statistic');
hold off
title(['Fig. 3. Simulated values exceeding the test statistic, N = ', ...
       num2str(N),', \nu = ',num2str(nu)]);
xlabel('t');
ylabel('Frequency');
legend('show');

% 4.3. Estimate the right-tail probability (simulated p-value)
simulatedPValue = numel(extremeValues)/numel(samples);

%% 5. Theoretical p-value calculation

% 5.1. Define the degrees of freedom
nu = 20;

% 5.2. Define evaluation points
xValues = -6:0.1:6;

% 5.3. Evaluate the PDF of the t-distribution
pdfValues = PDF(xValues,nu);

% 5.4. Plot the PDF and shade the p-value area
figure;
set(gcf,'Position',[100 100 1000 1000]);
plot(xValues,pdfValues,'-k','DisplayName','PDF');
hold on
xline(tValue,'-b','DisplayName','Test statistic');
tailRange = tValue:0.1:6;
tailDensity = pdf('T',tailRange,nu);
tailArea = area(tailRange,tailDensity,'DisplayName','P-value area');
tailArea.FaceColor = 'blue';
tailArea.EdgeColor = 'none';
hold off
title(['Fig. 4. Theoretical PDF of the t-distribution with shaded ' ...
    'p-value area',', \nu = ', num2str(nu)]);
xlabel('t');
ylabel('Density');
legend('show');

% 5.5. Compare the theoretical and simulated p-values
theoreticalPValue = 1 - cdf('T',tValue,nu);
