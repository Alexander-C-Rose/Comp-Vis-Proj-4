function log_li = log_likelihood(data, mu, Sigma, pi)
    % log likelihood function for Project 4
    [~, K] = size(mu); % tilde ignores the first output to conserve memory
    
    for k = 1:K % for each cluster, multiply alpha against the pdf
        temp(:,k) = pi(k) * mvnpdf(data, transpose(mu(:,k)), Sigma(:, :, k));
    end
    % Following the process of slide 6 Lecture 24, we sum, then log, then sum
    log_li = sum(log(sum(temp, 2)));
end