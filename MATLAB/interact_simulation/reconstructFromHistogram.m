function X = reconstructFromHistogram(histCounts)
% reconstructFromHistogram  Reconstruct a binary matrix from histogram counts
%
%   X = reconstructFromHistogram(histCounts)
%
%   Input:
%       histCounts : vector of length 2^m
%                    Counts for patterns in binary order 0:(2^m - 1)
%                    (000..0, 000..1, ..., 111..1)
%
%   Output:
%       X : m-by-N binary matrix, where N = sum(histCounts)

    % Validate input
    L = numel(histCounts);
    if L == 0 || (mod(log2(L),1) ~= 0)
        error('histCounts length must be a power of 2 (2^m).');
    end
    
    m = log2(L);        % number of rows
    N = sum(histCounts); % total number of columns
    
    X = zeros(m, N);    % Preallocate
    idx = 1;
    
    % Loop over each pattern
    for k = 0:(L-1)
        count = histCounts(k+1);
        if count > 0
            % Convert k to m-bit binary vector
            bits = dec2bin(k, m) - '0';
            % Repeat as needed
            X(:, idx:idx+count-1) = repmat(bits(:), 1, count);
            idx = idx + count;
        end
    end
    
    % Optional: shuffle column order
    X = X(:, randperm(N));
end
