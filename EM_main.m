clear;

load("kmeans.mat"); % K-means results
load("Normalized.mat"); % Gabor results post normalization and smoothing
truA = double(imread("mapA.bmp"));

% calculate alpha
% from k-means, what are the odds that a sample belongs to a given label?
% see slide 4 lecture 25

% kmeans initialization
choice = 3;
init_A = KA{choice};
kmeans_percent = accuracy(truA, KA{choice})
init_B = KA{best_IB};

[rA, cA] = size(init_A);
[rB, cB] = size(init_B);

% Find k-means label indices in similar form to gabor output
count = 0;
labsum = zeros(18,4);
for r = 1:rA
    for c = 1:cA
        for k = 1:4
            count = count + 1;
            if (init_A(r,c) == k)
                labsum(:,k) = labsum(:,k) + xA(count);
            end
        end
    end
end

% initialize values according to slide 4, lecture 25
for i = 1:4 % number of class labels
    temp = sum(init_A(:) == i);
    init_alpha_A(i) = temp/(rA*cA); 
    init_mu_A(:,i) = labsum(:,i) / temp;
    temp2 = xA - init_mu_A(i);
    init_sigma_A(:,:,i) = (temp2*transpose(temp2))/(temp-1); % Covariance
end



% EM algorithm

% tolerance for when log-likelihood function should stop EM algorithm
tol = 10;
% max number of iterations to run through (prevents the 
% program from never stopping if the tolerance isn't met
iterations = 50; 
for j = 1:iterations
    disp(j);
    % E-step
    for i = 1:4
        % take the multivariate distribution of the observation where rows are
        % data points and columns are the variable
        distribution = mvnpdf(transpose(xA), init_mu_A(i), init_sigma_A(:,:,i));
        gamma_I(:,i) = init_alpha_A(i) * distribution;
    end
    gamma_I = gamma_I ./ sum(gamma_I, 2);
    %disp(gamma_I(1:10,:));

    % M-step
    % This term is used multiple times
    gam_s = sum(gamma_I);
    alpha = gam_s ./ (rA*cA);
    mu = (xA * gamma_I) ./ gam_s;
    
%     % loop through clusters and calculate the new sigma parameter
%     for i = 1:4
%         temp = xA - init_mu_A(:,i);
%         sigma(:,:,i) = (temp * (gamma_I(:,i) .* transpose(temp))) / gam_s(i);
%     end
    % Loop through clusters and calculate the new sigma parameter
    for i = 1:4
        temp = xA - init_mu_A(:,i);
        sigma(:,:,i) = (temp * (gamma_I(:,i) .* transpose(temp))) / gam_s(i);
        
        % Enforce symmetry
        sigma(:,:,i) = (sigma(:,:,i) + sigma(:,:,i)') / 2;
        
        % Add a small positive value to the diagonal for regularization
        sigma(:,:,i) = sigma(:,:,i) + 1e-6 * eye(size(sigma(:,:,i)));
    end

    
    init_sigma_A = sigma;
    init_mu_A = mu;
    init_alpha_A = alpha;
    

    % log-likelihood function for given iteration
    log_out(j) = log_likelihood(transpose(xA), mu, sigma, alpha);


    % reshape gamma_I data for displaying
    count = 1;
    [M, label] = max(transpose(gamma_I));
    for r = 1:256
        for c = 1:256
            to_display(r,c,j) = label(count);
            count = count + 1;
        end
    end

    percent = accuracy(truA, to_display(:,:,j))

    % End the loop if the log_likelihood is within tolerance
    if j>1 && (abs(log_out(j) - log_out(j-1)) < tol)
        break;
    end
end


%% display results

figure(Color="White");
for i = 1:j-1
    subplot(1,j-1,i);
    imshow(mat2gray(to_display(:,:,i)));
end
