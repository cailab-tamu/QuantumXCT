import hlp.*
a = pwd;

cd('../manuscript/Dataset_8_PDGF_signaling_slows_ovarian_cancer GSM7019822_GSM7019823_GSM7019824/new_analysis/');
run('s0_merge_subunit_genes.m')
cd(a);

targettoplinkers = [7 2; 1 6; 3 5];
CellType1Genes = ["TGFB1", "PDGFRB", "IL6"];
CellType2Genes = ["STAT3","IL6RorST","TGFBR1or2","PDGFB"];
tags = ["Fibroblasts (Mo)", "Cancer Cells (Mo)", "Fibroblasts (Co)", "Cancer Cells (Co)"];
extragates = cxGate(4,5);

qmXct(sce, CellType1Genes, CellType2Genes, tags, targettoplinkers, extragates);

