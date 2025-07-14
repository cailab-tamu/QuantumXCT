%load 'lite_PDGF_signaling_slows_ovarian_cancer GSM7019822_GSM7019823_GSM7019824.mat'
%FibroblastGenes = ["TGFB1", "F3", "PDGFRB"];
%CancerGenes =     ["TGFBR2","SMAD3","HIF1A","PDGFB"];

run('new_analysis/s0_merge_subunit_genes.m')


FibroblastGenes = ["TGFB1", "PDGFRB", "IL6"];
CancerGenes = ["STAT3","IL6RorST","TGFBR1or2","PDGFB"];



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

theta = zeros(4 ,1);
layer_entg = [hGate(1:7); cnotGate(1,2); cnotGate(2,3); cnotGate(3,1);...
    cnotGate(4,5); cnotGate(5,6); cnotGate(6,7); cnotGate(7,4)];

layer_inte = [cryGate(5,4,theta(1)); ...
    cryGate(1,6,theta(2)); ...
    cryGate(7,2,theta(3)); ...
    cryGate(3,5,theta(4))];

C = quantumCircuit([layer_base; layer_entg; layer_inte]);


%%

methodid = 2;
switch methodid
    case 1
        options = optimset('Display','iter');
        [optimtheta, fval] = fminunc(@i_obj, [0 0 0 0], options, pt_f, pt_c, C);       
    case 2
        options = optimset('Display','iter');
        [optimtheta, fval] = fminsearch(@i_obj,[0 0 0 0], options,pt_f,pt_c,C);
    case 3
        options = optimoptions('fmincon','Display','iter');
        lb=-pi*ones(3,1);
        ub=pi*ones(3,1);
        [xa,fval] = fmincon(@i_obj,[0 0 0 0],[],[],[],[], ...
            lb,ub,[],options,pt_f,pt_c,C);        
end


figure;
nexttile
C.Gates(end).Angles = optimtheta(1);
C.Gates(end-1).Angles = optimtheta(2);
C.Gates(end-2).Angles = optimtheta(3);
C.Gates(end-3).Angles = optimtheta(4);
% C.Gates(end).Angles = 0;
% C.Gates(end-1).Angles = 0;
% C.Gates(end-2).Angles = 0;
% C.Gates(end-3).Angles = 0;

plot(C);
title(sprintf('Minimal total KL = %.3f\n(t_1 = %.3f, t_2 = %.3f, t_3 = %.3f, t_4 = %.3f)', ...
    fval, optimtheta(1), optimtheta(2), optimtheta(3), optimtheta(4)));

S = simulate(C);    
[states_f, po_f] = querystates(S,[1 2 3]);     % observed state pattern in fibroblast
[states_c, po_c] = querystates(S,[4 5 6 7]);   % observed state pattern in cancer   

% nexttile
% bar(pt_f)
% set(gca,'XTick',1:length(states_f));
% set(gca,'XTickLabel',states_f);
% ylabel('# of cells');
% xlabel('Expression pattern');
% title('Observed Pattern in Fibroblasts')
% 
% nexttile
% bar(po_f)
% set(gca,'XTick',1:length(states_f));
% set(gca,'XTickLabel',states_f);
% ylabel('# of cells');
% xlabel('Expression pattern');
% title('Quantum Simulation in Fibroblasts')
% 
% nexttile
% bar(pt_c)
% set(gca,'XTick',1:length(states_c));
% set(gca,'XTickLabel',states_c);
% ylabel('# of cells');
% xlabel('Expression pattern');
% title('Observed Pattern in Cancer Cells')

nexttile
bar([pt_c po_c e_patnfreq(f0_c)])
set(gca,'XTick',1:length(states_c));
set(gca,'XTickLabel',states_c);
ylabel('Freq. of cells');
xlabel('Expression pattern');
title('Cancer Cells')
legend({'Observed','Simulated','Theoretical'})

nexttile
bar([pt_f po_f e_patnfreq(f0_f)])
set(gca,'XTick',1:length(states_f));
set(gca,'XTickLabel',states_f);
ylabel('# of cells');
xlabel('Expression pattern');
title('Fibroblasts')
%legend({'Observed','Simulated','Theoretical'})



function [y] = i_obj(theta, pt_f, pt_c, C)
    C.Gates(end).Angles = theta(1);
    C.Gates(end-1).Angles = theta(2);
    C.Gates(end-2).Angles = theta(3);
    C.Gates(end-3).Angles = theta(4);
    S = simulate(C);
    [~, po_f] = querystates(S,[1 2 3]);     % observed state pattern in fibroblast
    [~, po_c] = querystates(S,[4 5 6 7]);   % observed state pattern in cancer   
    kl1 = i_kldiverg(pt_f, po_f);
    kl2 = i_kldiverg(pt_c, po_c);
    y = kl1+kl2;
end





function [layer] = in_12layers(f0)
    n = length(f0);
    layer1 = [];
    for k=1:n, layer1 = [layer1; ryGate(k,2*asin(sqrt(f0(k))))]; end
    % theta0 = pi*((rand(n,1)*2)-1);
    theta0 = zeros(n,1);
    layer2 = [];
    for k=1:n, layer2 = [layer2; rxGate(k, theta0(k))]; end
    % layer = [layer1; layer2];
    layer = [layer1];
end


