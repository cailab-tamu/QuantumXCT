function labels = e_label_propagation(A, max_iter)
    % A: Adjacency matrix (N x N)
    % max_iter: Maximum number of iterations
    % labels: Output community labels
%{
A = rand(500, 500) > 0.999;  % Example sparse adjacency matrix
A = triu(A,1); A = A + A';     % Make symmetric
labels = label_propagation(A, 10);
%}

    N = size(A, 1);                 % Number of nodes
    labels = (1:N)';                 % Initialize each node with a unique label
    
    for iter = 1:max_iter
        order = randperm(N);         % Random update order
        for i = order
            neighbors = find(A(i, :) > 0);
            if isempty(neighbors)
                continue;
            end
            % Get the most frequent label among neighbors
            neighbor_labels = labels(neighbors);
            unique_labels = unique(neighbor_labels);
            freq = histc(neighbor_labels, unique_labels);
            [~, idx] = max(freq);
            labels(i) = unique_labels(idx);
        end
    end
end
