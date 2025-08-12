function configsK = linkMatrix_dir_k(k,bag1,bag2)

if nargin<1, k = 3; end

% Define items in each bag:
%bag1 = {'A','B','C'};    % 3 items
%bag2 = {'1','2','3','4'}; % 4 items

if nargin<2, bag1 = {1,2,3};   end % 3 item
if nargin<3, bag2 = {4,5,6,7}; end % 4 items

% Build full list of directional links (24 total):
dirs = cell(numel(bag1) * numel(bag2) * 2, 2);
idx = 0;
for i = 1:numel(bag1)
    for j = 1:numel(bag2)
        idx = idx + 1;
        dirs{idx,1} = bag1{i};
        dirs{idx,2} = bag2{j};
        idx = idx + 1;
        dirs{idx,1} = bag2{j};
        dirs{idx,2} = bag1{i};
    end
end
% Tlinks = cell2table(dirs, 'VariableNames', {'From','To'});
Tlinks = cell2mat(dirs);

nLinks = height(Tlinks);  % = 24

% Choose exact k directed links:
% k = 3;

% Get all combinations of indices (each row is one subset of k indices):
idxComb = nchoosek(1:nLinks, k);  % from MATLAB docs :contentReference[oaicite:1]{index=1}
numC = size(idxComb, 1);

% Preallocate cell array to hold each k‑link configuration as a table
configsK = cell(numC, 1);

for t = 1:numC
    configsK{t} = Tlinks(idxComb(t, :), :);
end

% Example display for first few:
%{
for t = 1:min(5, numC)
    fprintf('Configuration %d (k=%d):\n', t, k);
    disp(configsK{t});
end
%}