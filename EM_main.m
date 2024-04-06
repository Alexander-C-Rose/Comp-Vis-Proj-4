clear;

load("kmeans.mat"); % K-means results
load("Normalized.mat"); % Gabor results post normalization and smoothing

% reshape gabor data for logical masking later
count = 1;
for i = 1:256
    for j = 1:256
        gab(i,j,:) = xA(:,count);
        count = count + 1;
    end
end

% calculate alpha
% from k-means, what are the odds that a sample belongs to a given label?
% see slide 4 lecture 25

% Good kmeans initialization
init_A = KA{best_I};
init_B = KA{best_IB};

[rA, cA] = size(init_A);
[rB, cB] = size(init_B);

% initialize values according to slide 4, lecture 25
for i = 1:4 % number of class labels
    temp = sum(init_A(:) == i);
    init_alpha_A(i) = temp/(rA*cA); 
    init_mu_A(i) = (sum(gab(init_A == i)))/temp;    % could maybe use centers?
    temp2 = xA - init_mu_A(i);
    init_sigma_A{i} = (temp2*transpose(temp2))/(temp-1); % Covariance
end



% EM algorithm
% A mosaic
iterations = 10;
%for i = 1:iterations
    % E-step
    for i = 1:4
        % take the multivariate distribution of the observation where rows are
        % data points and columns are the variable
        gamma_I(:,i) = init_alpha_A(i) * mvnpdf(transpose(xA), init_mu_A(i), init_sigma_A{i});
    end
    gamma_I = gamma_I ./ sum(gamma_I, 2);
%end

