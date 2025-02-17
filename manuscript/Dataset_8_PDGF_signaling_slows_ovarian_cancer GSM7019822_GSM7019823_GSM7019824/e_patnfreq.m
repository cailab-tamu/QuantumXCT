function [probs] = e_patnfreq(P_B)
    % Compute probabilities of drawing a white ball from each bag
    P_W = 1 - P_B;
    num_bags = length(P_B);
    
    % Generate all possible outcomes
    binary_combinations = dec2bin(0:(2^num_bags - 1)) - '0';
    outcomes = cell(size(binary_combinations, 1), 1);
    probs = zeros(size(binary_combinations, 1), 1);
    
    for i = 1:size(binary_combinations, 1)
        outcome = binary_combinations(i, :);
        outcomes{i} = strrep(num2str(outcome), ' ', '');
        probs(i) = prod(P_B(outcome == 1)) * prod(P_W(outcome == 0));
    end
    
    % Display results
    disp('Probabilities of all combinations:');
    for i = 1:length(outcomes)
        fprintf('P(%s) = %.3f\n', outcomes{i}, probs(i));
    end
    
    % Verify sum of probabilities
    sum_probs = sum(probs);
    fprintf('Sum of probabilities = %.3f (should be 1)\n', sum_probs);
end
