% Exercise - Halton sequences

%% 1. Aim of the exercise 
% Halton sequences are low-discrepancy, quasi-random sequences used for
% uniform sampling in high-dimensional spaces. This exercise illustrates
% their structure and behavior through visual comparisons with uniform and
% normal distributions, using both 2D and 3D plots.

%% 2. Theory
% Refer to the accompanying PDF file for the theory.

%% 3. Set the parameters of the Halton sequence

% 3.1. Clear the memory 
clear;

% 3.2. Set the number of observational units
N = 100;

% 3.3. Set the number of dimensions
dimensions = 1;

% 3.4. Set the number of draws per observational unit
draws = 1;

% 3.5. Define prime base for Halton sequence generation
prime = 3;

%% 4. Sampling with Halton sequences and uniform distribution

% 4.1. Generate a Halton sequence
[H,~] = halton(N,dimensions,draws, ...
    'prime',prime','leap',0,'scramble',0,'random',0,'burn',250);

% 4.2. Generate samples from a uniform(0,1) distribution
U = random("Uniform",0,1,[N,1]);

%% 5. Plot the Halton and uniform draws

% 5.1. Plot
figure
hold on
scatter(H,0.4*ones(N,1),'DisplayName',"Halton draws");
scatter(U,0.6*ones(N,1),'DisplayName',"Uniform draws");
axis([0 1 0 1])
legend('show')
title("Fig. 1. Comparison of Halton and uniform draws")
hold off

%% 6. Set the parameters of the Halton sequences

% 6.1. Clear the memory 
clear;

% 6.2. Set the number of observational units
N = 100;

% 6.3. Set the number of dimensions
dimensions = 2;

% 6.4. Set the number of draws per observational unit
draws = 1;

% 6.5. Define prime bases for Halton sequence generation
prime = [43,47];

%% 7. Halton sequences and transformation to standard normal dist.

% 7.1. Regular Halton
[H_standard_2D,Z_standard_2D] = halton(N,dimensions,draws, ...
    'prime',prime,'leap',0,'scramble',0,'random',0,'burn',250);

% 7.2. Randomized Halton
[H_random_2D,Z_random_2D] = halton(N,dimensions,draws, ...
    'prime',prime,'leap',0,'scramble',0,'random',1,'burn',250);

% 7.3. Scrambled Halton
[H_scramble_2D,Z_scramble_2D] = halton(N,dimensions,draws, ...
    'prime',prime,'leap',0,'scramble',1,'random',0,'burn',250);

% 7.4. Leap Halton
[H_leap_2D,Z_leap_2D] = halton(N,dimensions,draws, ...
    'prime',prime,'leap',4,'scramble',0,'random',0,'burn',250);

%% 8. Visual comparison of standard and scrambled Halton draws

% 8.1. Standard Halton sequences in 2D
figure
scatter(H_standard_2D(:,1),H_standard_2D(:,2), ...
    'DisplayName',"Standard Halton sequences")
ylabel("Dimension 2")
xlabel("Dimension 1")
legend('show')
title("Fig. 2. Standard Halton sequences")

% 8.2. Randomized Halton sequences in 2D
figure
scatter(H_random_2D(:,1),H_random_2D(:,2), ...
    'DisplayName',"Randomized Halton sequences")
ylabel("Dimension 2")
xlabel("Dimension 1")
legend('show')
title("Fig. 3. Randomized Halton sequences")

% 8.3. Scrambled Halton sequences in 2D
figure
scatter(H_scramble_2D(:,1),H_scramble_2D(:,2), ...
    'DisplayName',"Scrambled Halton sequences")
ylabel("Dimension 2")
xlabel("Dimension 1")
legend('show')
title("Fig. 4. Scrambled Halton sequences")

% 8.4. Leap Halton sequences in 2D
figure
scatter(H_leap_2D(:,1),H_leap_2D(:,2), ...
    'DisplayName',"Leap Halton sequences")
ylabel("Dimension 2")
xlabel("Dimension 1")
legend('show')
title("Fig. 5. Leap Halton sequences")

%% 9. Density estimation and 3D surface plot setup

% 9.1. Define grid for evaluating density surfaces
x1 = linspace(min(Z_standard_2D(:,1)),max(Z_standard_2D(:,1)),50);
x2 = linspace(min(Z_standard_2D(:,2)),max(Z_standard_2D(:,2)),50);
[X1,X2] = meshgrid(x1,x2);
grid_points = [X1(:),X2(:)];

% 9.2. Turn the transformed Halton draws into a density function
[f_standard,~] = ksdensity(Z_standard_2D,grid_points,'Function','pdf');
[f_random,~] = ksdensity(Z_random_2D,grid_points,'Function','pdf');
[f_scramble,~] = ksdensity(Z_scramble_2D,grid_points,'Function','pdf');
[f_leap,~] = ksdensity(Z_leap_2D,grid_points,'Function','pdf');

% 9.3. Reshape the densities for the surface plot
F_standard = reshape(f_standard,size(X1));
F_random = reshape(f_random,size(X1));
F_scrambled = reshape(f_scramble,size(X1));
F_leap = reshape(f_leap,size(X1));

% 9.4. Compute reference density from bivariate standard normal dist.
Z_reference = mvnpdf(grid_points);
Z_reference = reshape(Z_reference,size(X1));

%% 10. Compare Halton-based sampling densities to multivariate normal

% 10.1. Standard Halton vs. multivariate normal density
figure
hold on
surf(X1,X2,F_standard,'FaceColor','y','FaceAlpha',0.7, ...
    'DisplayName',"Standard Halton density");
surf(X1,X2,Z_reference,'FaceColor','m','FaceAlpha',0.1, ...
    'DisplayName',"Multivariate normal PDF");
ylabel('Y-axis');
xlabel('X-axis');
zlabel('Density');
legend('show');
title("Fig. 6. Standard Halton sampling vs. normal density");
view(3);
grid on
hold off

% 10.2. Randomized Halton vs. multivariate normal density
figure
hold on
surf(X1,X2,F_random,'FaceColor','y','FaceAlpha',0.7, ...
    'DisplayName',"Randomized Halton density");
surf(X1,X2,Z_reference,'FaceColor','m','FaceAlpha',0.1, ...
    'DisplayName',"Multivariate normal PDF");
ylabel('Y-axis');
xlabel('X-axis');
zlabel('Density');
legend('show');
title("Fig. 7. Randomized Halton sampling vs. normal density");
view(3);
grid on
hold off

% 10.3. Scrambled Halton vs. multivariate normal density
figure
hold on
surf(X1,X2,F_scrambled,'FaceColor','y','FaceAlpha',0.7, ...
    'DisplayName',"Scrambled Halton density");
surf(X1,X2,Z_reference,'FaceColor','m','FaceAlpha',0.1, ...
    'DisplayName',"Multivariate normal PDF");
ylabel('Y-axis');
xlabel('X-axis');
zlabel('Density');
legend('show');
title("Fig. 8. Scrambled Halton sampling vs. normal density");
view(3);
grid on
hold off

% 10.4. Leap Halton vs. multivariate normal density
figure
hold on
surf(X1,X2,F_leap,'FaceColor','y','FaceAlpha',0.7, ...
    'DisplayName',"Leap Halton density");
surf(X1,X2,Z_reference,'FaceColor','m','FaceAlpha',0.1, ...
    'DisplayName',"Multivariate normal PDF");
ylabel('Y-axis');
xlabel('X-axis');
zlabel('Density');
legend('show');
title("Fig. 9. Leap Halton sampling vs. normal density");
view(3);
grid on
hold off
