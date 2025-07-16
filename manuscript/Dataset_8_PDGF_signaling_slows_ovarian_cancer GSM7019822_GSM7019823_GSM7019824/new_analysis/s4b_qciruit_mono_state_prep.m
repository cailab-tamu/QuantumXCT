run('s0_merge_subunit_genes.m')
addpath ../
CancerGenes = ["STAT3","IL6RorST","TGFBR1or2","PDGFB"];
FibroblastGenes = ["TGFB1", "PDGFRB", "IL6"];

batquery = "Fibroblasts (Mo)";
genquery = FibroblastGenes;
[pt_f, ~, ~, f0_f] = getn(batquery, genquery, sce, false);

batquery = "Cancer Cells (Mo)";
genquery = CancerGenes;
[pt_c, ~, ~, f0_c] = getn(batquery, genquery, sce, false);



% [layer_base] = in_12layers([f0_f]);
    layer0 = [ryGate(1, 0.9395); ryGate(2, pi/4);...
        ryGate(3, pi/6); ...
        cxGate(1, 2); cxGate(2, 3); ...
        cxGate(3, 1)];


    layer1 = [ryGate(4, 1.9395); ryGate(5, pi/4);...
        ryGate(6, pi/6); ryGate(7, pi/8);...
        cxGate(4, 5); cxGate(5, 6); ...
        cxGate(6, 7); cxGate(7, 4)];

    % layer2 = rxGate(1:n, zeros(n,1));
    layer = [layer0; layer1];


C = quantumCircuit(layer);


%S = simulate(C);    
%[states_f, po_f] = querystates(S,[1 2 3]);     % observed state pattern in fibroblast
%[states_c, po_c] = querystates(S,[7 8 9 10]);   % observed state pattern in cancer   



methodid = 2;
switch methodid
    case 1
        options = optimset('Display','iter');
        [optimtheta, fval] = fminunc(@i_obj, [0 0 0 0 0 0 0], ...
            options, pt_f, pt_c, C);       
    case 2
        options = optimset('Display','iter');
        [optimtheta, fval] = fminsearch(@i_obj,[0 0 0 0 0 0 0], ...
            options, pt_f, pt_c, C);
    case 3
        options = optimoptions('fmincon','Display','iter');
        lb=-pi*ones(7,1);
        ub=pi*ones(7,1);
        [optimtheta,fval] = fmincon(@i_obj,[0 0 0 0 0 0 0],[],[],[],[], ...
            lb,ub,[],options,pt_f,pt_c,C);        
end

optimtheta

pause

C.Gates(1).Angles = optimtheta(1);
C.Gates(2).Angles = optimtheta(2);
C.Gates(3).Angles = optimtheta(3);
C.Gates(7).Angles = optimtheta(4);
C.Gates(8).Angles = optimtheta(5);
C.Gates(9).Angles = optimtheta(6);
C.Gates(10).Angles = optimtheta(7);

function [y] = i_obj(theta, pt_f, pt_c, C)
    a1 = [1 2 3];
    a2 = [7 8 9 10];
    a = [a1 a2];
    for k = 1:length(a)
        C.Gates(a(k)).Angles = theta(k);
    end

    S = simulate(C);
    [~, po_f] = querystates(S,[1 2 3]);     % observed state pattern in fibroblast
    [~, po_c] = querystates(S,[4 5 6 7]);   % observed state pattern in cancer   
pt_f
po_f
    kl1 = i_kldiverg(pt_f, po_f);
    kl2 = i_kldiverg(pt_c, po_c);
    y = kl1+kl2;
end




figure;
nexttile
plot(C)
nexttile
bar([pt_c po_c e_patnfreq(f0_c)])
set(gca,'XTick',1:length(states_c));
set(gca,'XTickLabel',states_c);
ylabel('Freq. of cells');
xlabel('Expression pattern');
title('Cancer Cells')

nexttile
bar([pt_f po_f e_patnfreq(f0_f)])
set(gca,'XTick',1:length(states_f));
set(gca,'XTickLabel',states_f);
ylabel('# of cells');
xlabel('Expression pattern');
title('Fibroblasts')
legend({'Observed','Simulated','Theoretical'})


