py.importlib.import_module('leidenalg');
py.importlib.import_module('igraph');

A = rand(500, 500) > 0.89;  % Example sparse adjacency matrix
A = triu(A,1); A = A + A';     % Make symmetric

% Convert adjacency matrix to edge list
[i, j] = find(triu(A)); 
edges = int16([i-1, j-1]);  % Python uses 0-based indexing

% Convert edges to Python list properly
edges_py = py.list(num2cell(edges', 1));
% Convert edges to Python list of tuples of integers
% edges_py = py.list(cellfun(@(row) py.tuple(py.int(row)), num2cell(edges, 2), 'UniformOutput', false));

% Create igraph graph object
G = py.igraph.Graph(py.int(size(A,1)), edges_py);



% Run Leiden algorithm
partition = py.leidenalg.find_partition(G, py.leidenalg.ModularityVertexPartition(G));
labels = double(cellfun(@double, cell(partition.membership))) + 1; % Convert back to MATLAB indexing



% Create a ModularityVertexPartition object
partition_type = py.leidenalg.ModularityVertexPartition(G);
% Find the partition
partition = py.leidenalg.find_partition(G, partition_type);


%%

% Create a simple graph
A = [0 1; 1 0];
edges = [1 2];
edges_zero_based = int16(edges - 1);


% Convert edges to Python list properly
edges_py = py.list(num2cell(edges_zero_based', 1));

% edges_py = py.list(cellfun(@(row) py.tuple(py.int(row)), num2cell(edges_zero_based, 2), 'UniformOutput', false));
G = py.igraph.Graph(py.int(size(A,1)), edges_py);

% Find partition
partition_type = py.leidenalg.ModularityVertexPartition(G);
partition = py.leidenalg.find_partition(G, partition_type);

% Display result
disp(partition.membership);