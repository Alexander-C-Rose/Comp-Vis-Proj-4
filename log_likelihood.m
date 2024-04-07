function log_li = log_likelihood(data, mu, Sigma, pi)
    [~, K] = size(mu);
    N = size(data, 1);
    
    log_li = 0;
    for k = 1:K
        log_li = log_li + sum(log(pi(k) * mvnpdf(data, transpose(mu(:,k)), Sigma(:, :, k))));
    end
    log_li = log_li / N;
end