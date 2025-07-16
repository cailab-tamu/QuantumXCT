run('s0_merge_subunit_genes.m')
addpath ../
CancerGenes = ["STAT3","IL6RorST","TGFBR1or2","PDGFB"];
FibroblastGenes = ["TGFB1", "PDGFRB", "IL6"];

batquery = "Fibroblasts (Mo)";
genquery = FibroblastGenes;
[pt_f_mo, ~, ~, f0_f_mo] = getn(batquery, genquery, sce, false);

batquery = "Cancer Cells (Mo)";
genquery = CancerGenes;
[pt_c_mo, ~, ~, f0_c_mo] = getn(batquery, genquery, sce, false);

batquery = "Fibroblasts (Co)";
genquery = FibroblastGenes;
[pt_f_co, ~, ~, f0_f_co] = getn(batquery, genquery, sce, false);

batquery = "Cancer Cells (Co)";
genquery = CancerGenes;
[pt_c_co, ~, ~, f0_c_co] = getn(batquery, genquery, sce, false);


amp = sqrt(pt_f_mo);                % unnormalized amplitudes vector
cg_f = initGate(1:3, amp);          % creates CompositeGate to initialize qubits 1–3 :contentReference[oaicite:1]{index=1}
qc_f = quantumCircuit(cg_f);        % initialize circuit with the gate
s_f = simulate(qc_f);                % simulate starting from |000>
[states_f, po_f_mo] = querystates(s_f, [1 2 3]); 

amp = sqrt(pt_c_mo);               % unnormalized amplitudes vector
cg_c = initGate(1:4, amp);         % creates CompositeGate to initialize qubits 1–3 :contentReference[oaicite:1]{index=1}
qc_c = quantumCircuit(cg_c);         % initialize circuit with the gate
s_c = simulate(qc_c);                % simulate starting from |000>
[states_c, po_c_mo] = querystates(s_c, [1 2 3 4]); 


cg1_mapped = compositeGate(cg_f, [1 2 3]);
cg2_mapped = compositeGate(cg_c, [4 5 6 7]);

theta = zeros(4 ,1);
layer_inte = [cryGate(5,4,theta(1)); ...
    cryGate(1,6,theta(2)); ...
    cryGate(7,2,theta(3)); ...
    cryGate(3,5,theta(4))];

combinedGate = [cg1_mapped; cg2_mapped; layer_inte];

C = quantumCircuit(combinedGate);

methodid = 3;
switch methodid
    case 1
        options = optimset('Display','iter');
        [optimtheta, fval] = fminunc(@i_obj, [0 0 0 0], options, pt_f_co, pt_c_co, C);       
    case 2
        options = optimset('Display','iter');
        [optimtheta, fval] = fminsearch(@i_obj,[0 0 0 0], options, pt_f_co, pt_c_co, C);
    case 3
        options = optimoptions('fmincon','Display','iter');
        lb=-pi*ones(4,1); ub=pi*ones(4,1);
        [optimtheta, fval] = fmincon(@i_obj,[0 0 0 0],[],[],[],[], ...
            lb, ub, [], options, pt_f_co, pt_c_co, C);        
end




S = simulate(C);    
[states_f, po_f_mo] = querystates(S,[1 2 3]);     % observed state pattern in fibroblast
[states_c, po_c_mo] = querystates(S,[4 5 6 7]);   % observed state pattern in cancer   


% optimtheta = randn(4 ,1);
C.Gates(end).Angles = optimtheta(1);
C.Gates(end-1).Angles = optimtheta(2);
C.Gates(end-2).Angles = optimtheta(3);
C.Gates(end-3).Angles = optimtheta(4);

S = simulate(C);    
[states_f2, po_f2] = querystates(S,[1 2 3]);     % observed state pattern in fibroblast
[states_c2, po_c2] = querystates(S,[4 5 6 7]);   % observed state pattern in cancer   


figure;
bar([pt_c_mo pt_c_co po_c2])
set(gca,'XTick',1:length(states_c));
set(gca,'XTickLabel',states_c);
ylabel('Freq. of cells');
xlabel('Expression pattern');
title('Cancer Cells')
legend({'Mono','Co','Simulated2'})



