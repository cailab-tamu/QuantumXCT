rng(123); % For reproducibility

% Parameters
nGenes = 50;
nCells = 300;
cellTypes = repelem({'A','B','C'},100);
genes = cell(nGenes,1);
for i = 1:nGenes
    genes{i} = ['Gene' num2str(i)];
end
% Add ligands/receptors
ligand1 = 'L1'; receptor1 = 'R1';
ligand2 = 'L2'; receptor2 = 'R2';
genes = [genes; {ligand1}; {receptor1}; {ligand2}; {receptor2}];
nGenesTotal = length(genes);

expr = poissrnd(2, nGenesTotal, nCells); % background low expression

% Programmed interactions:
% Type A: high L1
aIdx = strcmp(cellTypes, 'A');
expr(strcmp(genes, ligand1), aIdx) = poissrnd(20, 1, sum(aIdx));
% Type B: high R1
bIdx = strcmp(cellTypes, 'B');
expr(strcmp(genes, receptor1), bIdx) = poissrnd(20, 1, sum(bIdx));
% Type C: high L2/R2 (not programmed as interaction)
cIdx = strcmp(cellTypes, 'C');
expr(strcmp(genes, ligand2), cIdx) = poissrnd(20, 1, sum(cIdx));
expr(strcmp(genes, receptor2), cIdx) = poissrnd(20, 1, sum(cIdx));

% cellTypes: cell array (1 x nCells) of cell type assignment for each column
% genes: cell array (1 x nGenesTotal) of gene/lr names
% expr: matrix (nGenesTotal x nCells) single-cell counts

% Example: Display programmed ground truth
groundTruth.source = 'A';
groundTruth.target = 'B';
groundTruth.ligand = ligand1;
groundTruth.receptor = receptor1;

fprintf('Ground truth interaction: %s -> %s via %s-%s\n', ...
    groundTruth.source, groundTruth.target, ...
    groundTruth.ligand, groundTruth.receptor);

% For visualization:
% imagesc(expr); % Visualize the expression matrix

% Save to .mat or export to CSV if needed
% save('simExpr.mat','expr','genes','cellTypes','groundTruth');