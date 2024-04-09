function [to_display, log_out] = EM_function(gabor_data, kmeans, CA, cluster_num, tolerance, iterations)
%% This is the function version of EM_main.m
% It exists to provide a more generalized way to process the EM algorithm
% on data of different sizes with different feature vectors. 

% kmeans initialization
[rA, cA] = size(kmeans);

% This is used for the mu calculation
count = 0;
labsum = zeros(18,cluster_num);
for r = 1:rA
    for c = 1:cA
        for k = 1:cluster_num
            count = count + 1;
            if (kmeans(r,c) == k)
                labsum(:,k) = labsum(:,k) + gabor_data(count);
            end
        end
    end
end

% initialize values according to slide 4, lecture 25
for i = 1:cluster_num % number of class labels
    temp = sum(kmeans(:) == i);
    init_alpha_A(i) = temp/(rA*cA); 
    init_mu_A(:,i) = transpose(CA(i,:));
    temp2 = gabor_data - init_mu_A(:,i);
    init_sigma_A(:,:,i) = (temp2*transpose(temp2))/(temp-1); % Covariance
end

%% EM algorithm
for j = 1:iterations
    disp(j);

    % E-step
    for i = 1:cluster_num
        % take the multivariate distribution of the observation where rows are
        % data points and columns are the variable. gabor_data is a group
        % of column vectors, so it must be transposed. 

        % init_mu_A must have number of columns equal to the number of
        % columns of transpose(gabor_data) and in input to mvnpdf as a row
        % vector. Sigma is an 18 x 18 matrix when input to mvnpdf and is
        % the covariance matrix. 
        distribution = mvnpdf(transpose(gabor_data), transpose(init_mu_A(:,i)), init_sigma_A(:,:,i));
        gamma_I(:,i) = init_alpha_A(i) * distribution;
    end
    gam_sum = sum(gamma_I, 2);
    gamma_I = gamma_I ./ gam_sum;

    % M-step
    % This term is used multiple times
    gam_s = sum(gamma_I) + 0.000001; % adding small number to stop divide by zero
    alpha = gam_s ./ (rA*cA);
    mu = (gabor_data * gamma_I) ./ gam_s;

    fv_sz = size(gabor_data, 1); % size of the feature vector
    sigma = zeros(fv_sz, fv_sz, cluster_num); % initialize sigma

    % Loop through clusters and calculate the new sigma parameter
    for i = 1:cluster_num
        temp = gabor_data - init_mu_A(:,i);
        % multiply each vector (column vectors) by its probability
        % multiply this result by the transpose
        sigma(:,:,i) = ((transpose(gamma_I(:,i)) .* temp) * transpose(temp))/ gam_s(i);
    end

    % update parameters
    init_sigma_A = sigma;
    init_mu_A = mu;
    init_alpha_A = alpha;

    % reshape gamma_I data for displaying
    count = 1;
    [~, label] = max(transpose(gamma_I));
    for r = 1:rA
        for c = 1:cA
            to_display(r,c,j) = label(count);
            count = count + 1;
        end
    end
    
    % log-likelihood function for given iteration
    log_out(j) = log_likelihood(transpose(gabor_data), mu, sigma, alpha);
    
    % End the loop if the log_likelihood is within tolerance
    if j>1 && (abs(log_out(j) - log_out(j-1)) < tolerance)
        break;
    end
end
end % Function end