% improved by adding intracellular link between ["STAT3","IL6RorST"] see
% line 63
a = pwd;
targettop = [7 2; 1 6; 3 5];
cd('../manuscript/Dataset_8_PDGF_signaling_slows_ovarian_cancer GSM7019822_GSM7019823_GSM7019824/new_analysis/');
run('s0_merge_subunit_genes.m')
% addpath ./
% addpath ../
cd(a);
%%
FibroblastGenes = ["TGFB1", "PDGFRB", "IL6"];
CancerGenes = ["STAT3","IL6RorST","TGFBR1or2","PDGFB"];
FCGenes = [FibroblastGenes CancerGenes];

batquery = "Fibroblasts (Mo)"; genquery = FibroblastGenes;
[pt_f_mo, X_f_mo, ~, f0_f_mo] = hlp.getn(batquery, genquery, sce, false, true);

batquery = "Cancer Cells (Mo)"; genquery = CancerGenes;
[pt_c_mo, X_c_mo, ~, f0_c_mo] = hlp.getn(batquery, genquery, sce, false, true);

batquery = "Fibroblasts (Co)"; genquery = FibroblastGenes;
[pt_f_co, X_f_co, ~, ~] = hlp.getn(batquery, genquery, sce, false, true);

batquery = "Cancer Cells (Co)"; genquery = CancerGenes;
[pt_c_co, X_c_co, ~, ~] = hlp.getn(batquery, genquery, sce, false, true);

%%
% idx = ismember(sce.g, FCGenes);
% sce.g = sce.g(idx);
% sce.X = sce.X(idx, :);

scgeatoolApp(sce)

[y, idx]=ismember(sce1.c_cell_id, sce.c_cell_id);
assert(all(y))
assert(isequal(sce.c_cell_id(idx), sce1.c_cell_id))
ix = sce1.g=="IL6RorST";
sce1.X = [sce1.X; sce.X(ix, idx)];
sce1.g = [sce1.g; "IL6RorST"];
idx = ismember(sce1.g, FCGenes);
assert(sum(idx)==7)




