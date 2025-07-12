addpath ../
load '../lite_PDGF_signaling_slows_ovarian_cancer GSM7019822_GSM7019823_GSM7019824.mat'


x1 = sce.X(sce.g=="IL6R",:) + sce.X(sce.g=="IL6ST",:);
x2 = sce.X(sce.g=="TGFBR1",:) + sce.X(sce.g=="TGFBR2",:);
sce.g = [sce.g; "IL6RorST"; "TGFBR1or2"];
sce.X = [sce.X; x1; x2];

CancerGenes0= ["IL6RorST", "TGFBR1or2"];

