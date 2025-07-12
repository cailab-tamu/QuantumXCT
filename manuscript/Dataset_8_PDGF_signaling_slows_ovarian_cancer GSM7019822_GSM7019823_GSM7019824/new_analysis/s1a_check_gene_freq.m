%addpath ../
%load '../lite_PDGF_signaling_slows_ovarian_cancer GSM7019822_GSM7019823_GSM7019824.mat'
run('s0_merge_subunit_genes.m')



CancerGenes1 = ["HBEGF","EGFR","MAPK1","MAPK3","MAPK8","MAPK14","JUN","FOS"]; % https://www.cell.com/iscience/fulltext/S2589-0042(24)01860-1
CancerGenes2 = ["TGFBR2","SMAD3","HIF1A","PDGFB","IL6R","LIFR"];
CancerGenes3 = ["MET","FGFR1","FGFR2","CXCR4","ACKR3","IL6R","IL6ST","TGFBR1","TGFBR2","PDGFRA","PDGFRA","PDGFRB","PDGFRB","FLT1","KDR","FZD2","FZD5","ITGA5","ITGB1","CD44","ITGAV","PDGFA","PDGFB","PDGFA","PDGFB","TGFB1","TGFB1","IL1B","SHH","SHH","TNF","TNF"];
CancerGenes = unique([CancerGenes0, CancerGenes1 CancerGenes2 CancerGenes3]);
[y] = ismember(CancerGenes, sce.g);
CancerGenes = CancerGenes(y);
testgenes = CancerGenes;
testbatch = "Cancer Cells (Co)";
[f0_co] = in_get_f0(sce, testbatch, testgenes);
testbatch = "Cancer Cells (Mo)";
[f0_mo] = in_get_f0(sce, testbatch, testgenes);


%{

FibroblastGenes1 = ["HGF","FGF2","FGF7","CXCL12","CXCL12","IL6","IL6","TGFB1","TGFB1","PDGFA","PDGFB","PDGFA","PDGFB","VEGFA","VEGFA","WNT5A","WNT5A","SPARC","SPARC","SPP1","SPP1","PDGFRA","PDGFRA","PDGFRB","PDGFRB","TGFBR1","TGFBR2","IL1R1","PTCH1","SMO","TNFRSF1A","TNFRSF1B"];
FibroblastGenes2 = ["TGFB1", "IL6", "PDGFRB", "F3"];
FibroblastGenes3 = ["HBEGF","EGFR","MAPK1","MAPK3","MAPK8","MAPK14","JUN","FOS"];  % https://www.cell.com/iscience/fulltext/S2589-0042(24)01860-1
FibroblastGenes = unique([FibroblastGenes FibroblastGenes FibroblastGenes]);
[y] = ismember(FibroblastGenes, sce.g);
FibroblastGenes = FibroblastGenes(y);
testgenes = FibroblastGenes;
testbatch = "Fibroblasts (Co)";
[f0_co] = in_get_f0(sce, testbatch, testgenes);
testbatch = "Fibroblasts (Mo)";
[f0_mo] = in_get_f0(sce, testbatch, testgenes);
%}



%{
figure;
nexttile
[patn, X] = e_getnX(testbatch, testgenes, sce, false);
assert(sum(patn)==size(X,2))
f0 = sum(X,2)./size(X,2);         % per gene initial activate freq.
pt = patn./sum(patn);             % target cell state freq.    
nexttile
bar(f0)
set(gca,'XTick',1:length(f0));
set(gca,'XTickLabel', testgenes);
title('Frequency of Gene')
f0_co = f0;
%}




%{
figure;
nexttile
[patn, X] = e_getnX(testbatch, testgenes, sce, false);
assert(sum(patn)==size(X,2))
f0 = sum(X,2)./size(X,2);         % per gene initial activate freq.
pt = patn./sum(patn);             % target cell state freq.    
nexttile
bar(f0)
set(gca,'XTick',1:length(f0));
set(gca,'XTickLabel', testgenes);
title('Frequency of Gene')

f0_mo = f0;
%}

% idx = (f0>0.2 & f0<0.9);

f0diff = f0_co-f0_mo;
[~,idx] = sort(abs(f0diff),'descend');
sortedgenes = testgenes(idx)';
f0diff = f0diff(idx);
f0_co = f0_co(idx);
f0_mo = f0_mo(idx);
T = table(sortedgenes, f0diff, f0_co, f0_mo);



function [f0] = in_get_f0(sce, testbatch, testgenes)
    idxc = sce.c_batch_id == testbatch;
    [y, idxg] = ismember(testgenes, sce.g);
    assert(all(y));
    X = 0+(sce.X(idxg, idxc)>0);
    f0 = sum(X,2)./size(X,2);
end

