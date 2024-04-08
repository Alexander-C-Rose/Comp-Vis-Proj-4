clear;

load("kmeans.mat"); % K-means results
load("Normalized.mat"); % Gabor results post normalization and smoothing
truA = double(imread("mapA.bmp"));

% calculate alpha
% from k-means, what are the odds that a sample belongs to a given label?
% see slide 4 lecture 25

% kmeans initialization
choice = 1;
init_A = KB{choice};
gabor_data = xB;
k_per = perB(choice)

[rA, cA] = size(init_A);

% Find k-means label indices in similar form to gabor output
count = 0;
clusters = 3;

% This is used for the mu calculation
labsum = zeros(18,clusters);
for r = 1:rA
    for c = 1:cA
        for k = 1:clusters
            count = count + 1;
            if (init_A(r,c) == k)
                labsum(:,k) = labsum(:,k) + gabor_data(count);
            end
        end
    end
end

% initialize values according to slide 4, lecture 25
for i = 1:clusters % number of class labels
    temp = sum(init_A(:) == i);
    init_alpha_A(i) = temp/(rA*cA); 
    init_mu_A(:,i) = labsum(:,i) / temp;
    temp2 = gabor_data - init_mu_A(i);
    init_sigma_A(:,:,i) = (temp2*transpose(temp2))/(temp-1); % Covariance
end

%% EM algorithm

% tolerance for when log-likelihood function should stop EM algorithm
tol = 10;
% max number of iterations to run through (prevents the 
% program from never stopping if the tolerance isn't met
iterations = 50; 
for j = 1:iterations
    disp(j);

    % E-step
    for i = 1:clusters
        % take the multivariate distribution of the observation where rows are
        % data points and columns are the variable. gabor_data is a group
        % of column vectors, so it must be transposed. 

        % init_mu_A must have number of columns equal to the number of
        % columns of transpose(gabor_data) and in input to mvnpdf as a row
        % vector. Sigma is an 18 x 18 matrix when input to mvnpdf and is
        % the covariance matrix. 
        distribution = mvnpdf(transpose(gabor_data), init_mu_A(i), init_sigma_A(:,:,i));
        gamma_I(:,i) = init_alpha_A(i) * distribution;
    end
    gamma_I = gamma_I ./ sum(gamma_I, 2);

    % M-step
    % This term is used multiple times
    gam_s = sum(gamma_I) + 0.000001; % adding small number to stop divide by zero
    alpha = gam_s ./ (rA*cA);
    mu = (gabor_data * gamma_I) ./ gam_s;

    fv_sz = size(gabor_data, 1); % size of the feature vector
    sigma = zeros(fv_sz, fv_sz, clusters); % initialize sigma

    % Loop through clusters and calculate the new sigma parameter
    for i = 1:clusters
        temp = gabor_data - mu(:,i);
        % multiply each vector (column vectors) by its probability
        % multiply this result by the transpose
        sigma(:,:,i) = ((transpose(gamma_I(:,i)) .* temp) * transpose(temp))/ gam_s(i);
    end
    % loop through clusters and calculate the new sigma parameter
%     count = 0;
%     for i = 1:clusters
%         temp = gabor_data - init_mu_A(:,i);
%         for datapoint = 1:(rA*cA)
%             count = count+1
%             sigma(:,:,i) = sigma(:,:,i) + ((gamma_I(datapoint, i) ...
%                 .* (temp*transpose(temp))) / gam_s(i));
%         end
%     end
    % Loop through clusters and calculate the new sigma parameter
%     for i = 1:clusters
%         temp = gabor_data - mu(:,i);
%         sigma(:,:,i) = (temp * (gamma_I(:,i) .* transpose(temp))) / gam_s(i);
%         
%         % Enforce symmetry
%         sigma(:,:,i) = (sigma(:,:,i) + sigma(:,:,i)') / 2;
%         
%         % Add a small positive value to the diagonal for regularization
%         sigma(:,:,i) = sigma(:,:,i) + 1e-6 * eye(size(sigma(:,:,i)));
%     end

    % update parameters
    init_sigma_A = sigma;
    init_mu_A = mu;
    init_alpha_A = alpha;
    

    % log-likelihood function for given iteration
    log_out(j) = log_likelihood(transpose(gabor_data), mu, sigma, alpha);


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
for i = 1:8
    subplot(2,4,i);
    imshow(mat2gray(to_display(:,:,i)));
end
