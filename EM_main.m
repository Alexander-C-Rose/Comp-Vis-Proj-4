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
    init_sigma_A(:,:,i) = (temp2*transpose(temp2))/(temp-1); % Covariance
end



% EM algorithm

% tolerance for when log-likelihood function should stop EM algorithm
tol = 5;
% max number of iterations to run through (prevents the 
% program from never stopping if the tolerance isn't met
iterations = 50; 
for j = 1:iterations
    disp(j);
    % E-step
    for i = 1:4
        % take the multivariate distribution of the observation where rows are
        % data points and columns are the variable
        gamma_I(:,i) = init_alpha_A(i) * mvnpdf(transpose(xA), init_mu_A(i), init_sigma_A(:,:,i));
    end
    gamma_I = gamma_I ./ sum(gamma_I, 2);
    disp(gamma_I(1:10,:));

    % M-step
    % new alpha is the number of data points assigned to a label divided by
    % total data points.
    gam_s = sum(gamma_I); % This term is used multiple times; precalc. here
    alpha = gam_s / (rA*cA);
    mu = (transpose(gamma_I) * transpose(xA)) ./ transpose(gam_s);

    % loop through clusters and calculate the new sigma parameter
    for i = 1:4
        temp = transpose(xA) - mu(i,:);
        sigma(:,:,i) = (transpose(temp) * (gamma_I(:,i) .* temp)) / gam_s(i);
    end
    
    init_sigma_A = sigma;
    init_mu_A = mu;
    init_alpha_A = alpha;

    % log-likelihood function for given iteration
    for i = 1:4
        log_l(:,i) = alpha(i) * mvnpdf(transpose(xA), mu(i,:), sigma(:,:,i));
    end
    log_likelihood(j) = sum(log(sum(log_l)));
    % reshape gamma_I data for displaying
    count = 1;
    [M, label] = max(transpose(gamma_I));
    for r = 1:256
        for c = 1:256
            to_display(r,c,j) = label(count);
            count = count + 1;
        end
    end
    % End the loop if the log_likelihood is within tolerance
    if j>1 && (abs(log_likelihood(j) - log_likelihood(j-1)) < tol)
        break;
    end
end


% reshape gamma_I data for displaying
count = 1;
[M, label] = max(transpose(gamma_I));
for i = 1:256
    for j = 1:256
        to_display(i,j,:) = label(count);
        count = count + 1;
    end
end

figure(Color="White")
imshow(mat2gray(to_display(:,:,1)));
