% Number of items in Bag 1 and Bag 2
n1 = 3;
n2 = 4;

numLinks = n1 * n2;
numComb = 2^numLinks;

bag1_labels = {'A','B','C'};
bag2_labels = {'1','2','3','4'};

for idx = 0:numComb-1
    % Get binary vector for current combination
    bits = bitget(idx, 1:numLinks);
    % Reshape to 3x4 "link matrix"
    linkmat = reshape(bits, n2, n1)';
    
    % Print the links for this combination
    fprintf('Combination %d:\n', idx+1);
    [r,c] = find(linkmat);
    if isempty(r)
        disp('  No links.');
    else
        for k = 1:numel(r)
            fprintf('  Link: %s - %s\n', bag1_labels{r(k)}, bag2_labels{c(k)});
        end
    end
    fprintf('\n');
    
    % Uncomment below to display the matrix as well
    % disp(linkmat);
    pause
end