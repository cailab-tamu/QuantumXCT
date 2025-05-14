load('lite_PDGF_signaling_slows_ovarian_cancer GSM7019822_GSM7019823_GSM7019824.mat','sce');

FibroblastGenes = ["TGFB1", "F3", "PDGFRB"];
CancerGenes =     ["TGFBR2","SMAD3","HIF1A","PDGFB"];

N = 7;

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
layer_inte = [  cryGate(1,4,theta(1)); ...
                cryGate(1,5,theta(2)); ...
                cryGate(7,3,theta(3));...
                cryGate(6,2,theta(4))];
C = quantumCircuit([layer_base; layer_inte]);

plot(C)

%%
methodid = 1;
initheta = zeros(1, 4);
switch methodid
    case 1
        options = optimset('Display','iter');
        [optimtheta, fval] = fminunc(@i_obj, initheta, options, pt_f, pt_c, C);       
    case 2
        options = optimset('Display','iter');
        [optimtheta, fval] = fminsearch(@i_obj,initheta, options,pt_f,pt_c,C);
    case 3
        options = optimoptions('fmincon','Display','iter');
        lb=-pi*ones(4,1);
        ub=pi*ones(4,1);
        [xa,fval] = fmincon(@i_obj,initheta,[],[],[],[], ...
            lb,ub,[],options,pt_f,pt_c,C);        
end


figure;
nexttile
    C.Gates(N+1).Angles = optimtheta(1);
    C.Gates(N+2).Angles = optimtheta(2);
    C.Gates(N+3).Angles = optimtheta(3);
    C.Gates(N+4).Angles = optimtheta(4);
%plot(C);
%title(sprintf('Minimal total KL = %.3f\n(t_1 = %.3f, t_2 = %.3f, t_3 = %.3f)', ...
%    fval, optimtheta(1), optimtheta(2), optimtheta(3)));

S = simulate(C);    
[states_f, po_f] = querystates(S,[1 2 3]);     % observed state pattern in fibroblast
[states_c, po_c] = querystates(S,[4 5 6 7]);   % observed state pattern in cancer   

bar(pt_f)
set(gca,'XTick',1:length(states_f));
set(gca,'XTickLabel',states_f);
ylabel('# of cells');
xlabel('Expression pattern');
title('Observed Pattern in Fibroblasts (Co)')

% nexttile
% bar(po_f)
% set(gca,'XTick',1:length(states_f));
% set(gca,'XTickLabel',states_f);
% ylabel('# of cells');
% xlabel('Expression pattern');
% title('Quantum Simulation in Fibroblasts')

nexttile
bar(pt_c)
set(gca,'XTick',1:length(states_c));
set(gca,'XTickLabel',states_c);
ylabel('# of cells');
xlabel('Expression pattern');
title('Observed Pattern in Cancer Cells (Co)')

% nexttile
% bar([pt_c po_c e_patnfreq(f0_c)])
% set(gca,'XTick',1:length(states_c));
% set(gca,'XTickLabel',states_c);
% ylabel('Freq. of cells');
% xlabel('Expression pattern');
% title('Cancer Cells')
% legend({'Observed','Simulated','Theoretical'})

% nexttile
% bar([pt_f po_f e_patnfreq(f0_f)])
% set(gca,'XTick',1:length(states_f));
% set(gca,'XTickLabel',states_f);
% ylabel('# of cells');
% xlabel('Expression pattern');
% title('Fibroblasts')
% legend({'Observed','Simulated','Theoretical'})

nexttile
testbatch = "Fibroblasts (Mo)";
testgenes = FibroblastGenes;
[patn, X] = e_getnX(testbatch, testgenes, sce, false);
f0_f = sum(X,2)./size(X,2);         % per gene initial activate freq.
pt_f = patn./sum(patn);             % target cell state freq.
bar(pt_f)
set(gca,'XTick',1:length(states_f));
set(gca,'XTickLabel',states_f);
ylabel('# of cells');
xlabel('Expression pattern');
title('Observed Pattern in Fibroblasts (Mo)')

nexttile
testbatch = "Cancer Cells (Mo)";
testgenes = CancerGenes;
[patn, X] = e_getnX(testbatch, testgenes, sce, false);
f0_c = sum(X,2)./size(X,2);         % per gene initial activate freq.
pt_c = patn./sum(patn);             % target cell state freq.
bar(pt_c)
set(gca,'XTick',1:length(states_c));
set(gca,'XTickLabel',states_c);
ylabel('# of cells');
xlabel('Expression pattern');
title('Observed Pattern in Cancer Cells (Mo)')

function [y] = i_obj(theta, pt_f, pt_c, C)
    N = 7;
    C.Gates(N+1).Angles = theta(1);
    C.Gates(N+2).Angles = theta(2);
    C.Gates(N+3).Angles = theta(3);
    C.Gates(N+4).Angles = theta(4);
    S = simulate(C);    
    [~, po_f] = querystates(S,[1 2 3]);     % observed state pattern in fibroblast
    [~, po_c] = querystates(S,[4 5 6 7]);   % observed state pattern in cancer   
    kl1 = i_kldiverg(pt_f, po_f);
    kl2 = i_kldiverg(pt_c, po_c);
    y = kl1+kl2;
end

function [layer] = in_12layers(f0)
    n = length(f0);
    layer1 = ryGate(1:n, 2*asin(sqrt(f0)));
    % layer2 = rxGate(1:n, zeros(n,1));
    layer = [layer1];
end


