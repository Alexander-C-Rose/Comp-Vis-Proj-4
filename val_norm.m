function [norm_val] = val_norm(value, min_val, max_val)
% This function is a simple normalization equation. The output will be
% within the range of [0 1].

norm_val = (value - min_val) ./ (max_val - min_val);

end