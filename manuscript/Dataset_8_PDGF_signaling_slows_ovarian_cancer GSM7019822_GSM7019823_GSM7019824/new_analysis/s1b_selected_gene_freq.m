%addpath ../
%load '../lite_PDGF_signaling_slows_ovarian_cancer GSM7019822_GSM7019823_GSM7019824.mat'
run('s0_merge_subunit_genes.m')

CancerGenes = ["STAT3","IL6RorST","TGFBR1or2","PDGFB"];
FibroblastGenes = ["TGFB1", "PDGFRB", "IL6"];


testgenes = CancerGenes;
testbatch = "Cancer Cells (Co)";
[f0_co] = in_get_f0(sce, testbatch, testgenes);
testbatch = "Cancer Cells (Mo)";
[f0_mo] = in_get_f0(sce, testbatch, testgenes);


%{
testgenes = FibroblastGenes;
testbatch = "Fibroblasts (Co)";
[f0_co] = in_get_f0(sce, testbatch, testgenes);
testbatch = "Fibroblasts (Mo)";
[f0_mo] = in_get_f0(sce, testbatch, testgenes);
%}


figure;
nexttile
bar(f0_co)
set(gca,'XTick',1:length(f0_co));
set(gca,'XTickLabel', testgenes);
title('Co')

nexttile
bar(f0_mo)
set(gca,'XTick',1:length(f0_mo));
set(gca,'XTickLabel', testgenes);
title('Mo')

nexttile
bar(f0_co-f0_mo)
set(gca,'XTick',1:length(f0_co));
set(gca,'XTickLabel', testgenes);
title('Co-Mo')


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

