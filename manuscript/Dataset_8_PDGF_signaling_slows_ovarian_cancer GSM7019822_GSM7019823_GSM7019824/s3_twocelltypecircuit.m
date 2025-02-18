load 'lite_PDGF_signaling_slows_ovarian_cancer GSM7019822_GSM7019823_GSM7019824.mat'

FibroblastGenes = ["TGFB1", "IL6", "PDGFRB"];
CancerGenes =     ["TGFBR2","SMAD3","HIF1A","PDGFB"];


testbatch = "Fibroblasts (Co)";
testgenes = FibroblastGenes;
[patn, X] = e_getnX(testbatch, testgenes, sce, false);
f0_f = sum(X,2)./size(X,2);         % per gene initial activate freq.
pt_f = patn./sum(patn);             % target cell state freq.

testbatch = "Cancer Cells (Co)";
testgenes = CancerGenes;
[patn, X] = e_getnX(testbatch, testgenes, sce, false);
f0_c = sum(X,2)./size(X,2);         % per gene initial activate freq.
pt_c = patn./sum(patn);             % target cell state freq.

[layer_base] = in_12layers([f0_f; f0_c]);

theta = zeros(3 ,1);
layer_inte = [cryGate(1,4,theta(1)); ...
    cryGate(1,5,theta(2)); ...
    cryGate(7,3,theta(3))];
C = quantumCircuit([layer_base; layer_inte]);
figure; 
plot(C);

%%

for rep = 1:5
    
    theta = randn(3, 1);
    C.Gates(15).Angles = theta(1);
    C.Gates(16).Angles = theta(2);
    C.Gates(17).Angles = theta(3);

    S = simulate(C);    
    [states_f, po_f] = querystates(S,[1 2 3]);     % observed state pattern in fibroblast
    [states_c, po_c] = querystates(S,[4 5 6 7]);   % observed state pattern in cancer
    
    kl1 = i_kldiverg(pt_f, po_f);
    kl2 = i_kldiverg(pt_c, po_c);
    fprintf('KL_f = %f, KL_c = %f, total KL = %f\n',kl1, kl2, kl1+kl2);
end





function [layer] = in_12layers(f0)
    n = length(f0);
    layer1 = [];
    for k=1:n, layer1 = [layer1; ryGate(k,2*asin(sqrt(f0(k))))]; end
    % theta0 = pi*((rand(n,1)*2)-1);
    theta0 = zeros(n,1);
    layer2 = [];
    for k=1:n, layer2 = [layer2; rxGate(k, theta0(k))]; end
    layer = [layer1; layer2];
end


