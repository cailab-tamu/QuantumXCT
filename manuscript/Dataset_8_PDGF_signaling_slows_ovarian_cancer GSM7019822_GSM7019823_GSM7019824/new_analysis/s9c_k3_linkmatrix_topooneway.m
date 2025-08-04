run('s0_merge_subunit_genes.m')
addpath ../
FibroblastGenes = ["TGFB1", "PDGFRB", "IL6"];
CancerGenes = ["STAT3","IL6RorST","TGFBR1or2","PDGFB"];

batquery = "Fibroblasts (Mo)"; genquery = FibroblastGenes;
[pt_f_mo, ~, ~, f0_f_mo] = getn(batquery, genquery, sce, false);

batquery = "Cancer Cells (Mo)"; genquery = CancerGenes;
[pt_c_mo, ~, ~, f0_c_mo] = getn(batquery, genquery, sce, false);

batquery = "Fibroblasts (Co)"; genquery = FibroblastGenes;
[pt_f_co, ~, ~, ~] = getn(batquery, genquery, sce, false);

batquery = "Cancer Cells (Co)"; genquery = CancerGenes;
[pt_c_co, ~, ~, ~] = getn(batquery, genquery, sce, false);

%% 
% quantum state preparation via uniformly controlled rotations
amp = sqrt(pt_f_mo);                % unnormalized amplitudes vector
% log2(numel(pt_f_mo)) = 3
cg_f = initGate(1:3, amp);          % creates CompositeGate to initialize qubits 1–3 :contentReference[oaicite:1]{index=1}
qc_f = quantumCircuit(cg_f);        % initialize circuit with the gate

amp = sqrt(pt_c_mo);                % unnormalized amplitudes vector
cg_c = initGate(1:4, amp);          % creates CompositeGate to initialize qubits 1–4 :contentReference[oaicite:1]{index=1}
qc_c = quantumCircuit(cg_c);        % initialize circuit with the gate


%%
cg1_mapped = compositeGate(cg_f, [1 2 3]);
cg2_mapped = compositeGate(cg_c, [4 5 6 7]);

combinedGate = [cg1_mapped; cg2_mapped];
C = quantumCircuit(combinedGate);
S = simulate(C);    
[states_f, po_f_mo] = querystates(S,[1 2 3]);     % observed state pattern in fibroblast
[states_c, po_c_mo] = querystates(S,[4 5 6 7]);   % observed state pattern in cancer   


% layer_entg = [cxGate(1, 2); cxGate(2, 3); cxGate(3, 1);...
%     cxGate(4, 5); cxGate(5, 6); cxGate(6, 7);...
%     cxGate(7, 4)];
% [layer_base] = in_12layers([f0_f_mo; f0_c_mo]);

%%
configsK = linkMatrix_dir_k(3);
numComb = length(configsK);

Y = nan(numComb, 1);
Cc = cell(numComb, 1);

for idx = 1:numComb


    if any(ismember(fliplr(configsK{idx}), configsK{idx}, "rows")), continue; end    

    layer_inte = [];
    for k = 1:height(configsK{idx})
        layer_inte = [layer_inte; cxGate(configsK{idx}(k,1), configsK{idx}(k,2))];
    end
    combinedGate = [cg1_mapped; cg2_mapped; layer_inte];
    C = quantumCircuit(combinedGate);
    Cc{idx} = C;
    S = simulate(C);
    [~, po_f] = querystates(S,[1 2 3]);     % observed state pattern in fibroblast
    [~, po_c] = querystates(S,[4 5 6 7]);   % observed state pattern in cancer   
    kl1 = i_kldiverg(pt_f_co, po_f);
    kl2 = i_kldiverg(pt_c_co, po_c);
    Y(idx) = kl1 + kl2;
    fprintf('Combination %d: %f\n', idx, Y(idx));
end

%%
[a, b] = mink(Y, 10);
figure;
plot(Y);

FibroblastGenes = ["TGFB1", "PDGFRB", "IL6"];
CancerGenes = ["STAT3","IL6RorST","TGFBR1or2","PDGFB"];
FCGenes = [FibroblastGenes CancerGenes];


for topk = 1:1


    C = Cc{b(topk)};
    S = simulate(C);
    [states_f2, po_f2] = querystates(S,[1 2 3]);     % observed state pattern in fibroblast
    [states_c2, po_c2] = querystates(S,[4 5 6 7]);   % observed state pattern in cancer   
    
    figure;
    nexttile
    plot(C)
    
    nexttile
    for k = 1:length(configsK{b(topk)})
        x = configsK{b(topk)}(k, 1) +" "+ FCGenes(configsK{b(topk)}(k, 1))+...
            " -> "+configsK{b(topk)}(k, 2)+" "+FCGenes(configsK{b(topk)}(k, 2));
        text(0, 50 - k*10, x);
        ylim([0 50]);
        axis off
    end

    nexttile
    bar([pt_c_mo pt_c_co po_c2])
    set(gca,'XTick',1:length(states_c));
    set(gca,'XTickLabel',states_c);
    ylabel('Freq. of cells');
    xlabel('Expression pattern');
    title('Cancer Cells')
    %legend({'Mono','Co','Simulated2'})
    
    nexttile
    bar([pt_f_mo pt_f_co po_f2])
    set(gca,'XTick',1:length(states_f));
    set(gca,'XTickLabel',states_f);
    ylabel('Freq. of cells');
    xlabel('Expression pattern');
    title('Fibroblasts')
    %legend({'Mono','Co','Simulated2'})



end