% Exercise - Understanding hypothesis tesing using simulation

%% 1. Clear workspace

% Clear
clear;

%% 2. Define t value and degrees of freedom

% 2.1. Define t value
tValue = 1.8;

% 2.2. Define degrees of freedom
nu = 20;

%% 3. Generate Student's t distributed values

% 3.1. Number of random draws
N = 20000;

% 3.2. Define histogram bin edges
binEdges = -6:0.01:6;

% 3.3. Construct t-distributed random variables
draws = StochasticRepresentation(N,nu);

% 3.4. Create histogram
figure;
set(gcf,'Position',[100 100 1000 1000]);
histogram(draws,...
    'BinEdges',binEdges,...
    'FaceColor','#b3b3b3',...
    'EdgeAlpha',0,...
    'DisplayName','All simulated values');
title(['Fig. 1. Histogram of simulated t-distributed draws, N = ', ...
       num2str(N), ', \nu = ', num2str(nu)]);
xlabel('t');
ylabel('Frequency');
legend('show');

%% 4. Add the test statistic to the plot

% Plot histogram and test statistic
figure;
set(gcf,'Position',[100 100 1000 1000]);
histogram(draws,...
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

%% 5. Highlight values more extreme than the test statistic

% 5.1. Select values exceeding the test statistic
extremeValues = draws(draws > tValue);

% 5.2. Plot histogram and highlight values exceeding the test statistic
figure;
set(gcf,'Position',[100 100 1000 1000]);
histogram(draws,...
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

% 5.3. Estimate the right-tail probability (simulated p-value)
simulatedPValue = numel(extremeValues)/numel(draws);

%% 6. Theoretical p-value calculation

% 6.1. Define the degrees of freedom
nu = 20;

% 6.2. Define evaluation points
xValues = -6:0.1:6;

% 6.3. Evaluate the PDF of the t-distribution
pdfValues = PDF(xValues,nu);

% 6.4. Plot the PDF and shade the p-value area
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

% 6.5. Compare the theoretical and simulated p-values
theoreticalPValue = 1 - cdf('T',tValue,nu);
