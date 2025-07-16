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



amp = sqrt(pt_f);  % unnormalized amplitudes vector
cg_f = initGate(1:3, amp);         % creates CompositeGate to initialize qubits 1–3 :contentReference[oaicite:1]{index=1}
qc_f = quantumCircuit(cg_f);         % initialize circuit with the gate
s_f = simulate(qc_f);                % simulate starting from |000>
%disp(formula(s_f));                % display state formula :contentReference[oaicite:2]{index=2}
%histogram(s_f);                    % visualize probabilities over basis states
%p = probability(s_f, [1 2 3]);     % compute total probability (should be 1) :contentReference[oaicite:3]{index=3}
[states_f, po_f] = querystates(s_f, [1 2 3]); 


amp = sqrt(pt_c);  % unnormalized amplitudes vector
cg_c = initGate(1:4, amp);         % creates CompositeGate to initialize qubits 1–3 :contentReference[oaicite:1]{index=1}
qc_c = quantumCircuit(cg_c);         % initialize circuit with the gate
s_c = simulate(qc_c);                % simulate starting from |000>
%disp(formula(s_f));                % display state formula :contentReference[oaicite:2]{index=2}
%histogram(s_f);                    % visualize probabilities over basis states
%p = probability(s_f, [1 2 3]);     % compute total probability (should be 1) :contentReference[oaicite:3]{index=3}
[states_c, po_c] = querystates(s_c, [1 2 3 4]); 



figure;
nexttile
plot(qc_f)
nexttile
plot(qc_c)
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




