error('needs large memory.');


%{
Why this works:
Bidirectional generation: You loop through each bag1–bag2 pair and create two rows—one for each direction.
Full power‑set: With nLinks = 24, you generate all subsets of the 24 directed links (excluding the empty set)—i.e. you allow any combination of links, each direction counted separately.
Result: configs{} is a cell array, each cell containing a table listing the selected directed edges for that configuration. You get 
2^24−1≈16 million configurations.
%}

% Define items in each bag:
bag1 = {'A','B','C'};
bag2 = {'1','2','3','4'};

% Build full list of *directional* links:
% A→1, A→2, … ; B→1, … ; and also A←1, A←2, … ; i.e. treat each pair bidirectionally
n1 = numel(bag1);
n2 = numel(bag2);

% Preallocate
dirs = cell(n1 * n2 * 2, 2);   % each row: {From, To}
idx = 0;
for i = 1:n1
    for j = 1:n2
        idx = idx + 1;
        dirs{idx,1} = bag1{i};
        dirs{idx,2} = bag2{j};
        idx = idx + 1;
        dirs{idx,1} = bag2{j};
        dirs{idx,2} = bag1{i};
    end
end

% Convert to table
Tlinks = cell2table(dirs, 'VariableNames', {'From','To'});
nLinks = size(Tlinks,1);  % = 3×4×2 = 24
N = 2^nLinks - 1;          % all non-empty subsets

% Preallocate cell array of configurations:
configs = cell(N,1);

for k = 1:N
    mask = logical(bitget(k, 1:nLinks));
    configs{k} = Tlinks(mask,:);
end

% Example: display first few configurations
for i = 1:5
    fprintf('Config %d: using %d directed links\n', i, height(configs{i}));
    disp(configs{i});
end





%{
    k = 3;
    idxAll = 1:nLinks;
    combs = nchoosek(idxAll, k);
    numC = size(combs,1);
    configsK = cell(numC,1);
    for t = 1:numC
        configsK{t} = Tlinks(combs(t,:), :);
    end
%}
