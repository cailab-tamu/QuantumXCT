% improved by adding intracellular link between ["STAT3","IL6RorST"] see
% line 63
import hlp.*
a = pwd;
targettop = [7 2; 1 6; 3 5];
cd('../manuscript/Dataset_8_PDGF_signaling_slows_ovarian_cancer GSM7019822_GSM7019823_GSM7019824/new_analysis/');
run('s0_merge_subunit_genes.m')
cd(a);
%%
FibroblastGenes = ["TGFB1", "PDGFRB", "IL6"];
CancerGenes = ["STAT3","IL6RorST","TGFBR1or2","PDGFB"];
FCGenes = [FibroblastGenes CancerGenes];


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
configsK = hlp.linkMatrix_dir_k(4);
numComb = length(configsK);

Y = nan(numComb, 1);
Cc = cell(numComb, 1);
idealY = [];
isideal = false(numComb, 1);

for idx = 1:numComb

    if any(ismember(fliplr(configsK{idx}), configsK{idx}, "rows")), continue; end
    if length(unique(configsK{idx}))<6, continue; end

    layer_inte = [];
    for k = 1:height(configsK{idx})
        layer_inte = [layer_inte; cxGate(configsK{idx}(k,1), configsK{idx}(k,2))];
    end
    combinedGate = [cg1_mapped; cg2_mapped; cxGate(4, 5);  layer_inte];
    C = quantumCircuit(combinedGate);
    Cc{idx} = C;
    S = simulate(C);
    [~, po_f] = querystates(S,[1 2 3]);     % observed state pattern in fibroblast
    [~, po_c] = querystates(S,[4 5 6 7]);   % observed state pattern in cancer   
    kl1 = i_kldiverg(pt_f_co, po_f);
    kl2 = i_kldiverg(pt_c_co, po_c);
    Y(idx) = kl1 + kl2;
    fprintf('Combination %d: %f\n', idx, Y(idx));
    if sum(ismember(sort(configsK{idx},2), sort(targettop, 2),"rows"))>=3
        idealY = [idealY; Y(idx)];
        isideal(idx) = true;
    end

end

%%
[a, b] = mink(Y, 10);
figure;
plot(Y);
yline(idealY,'r');



%%
for topk = 1:3
    t = hlp.fun_drawreshisto(b(topk), Cc, configsK, ...
        pt_c_mo, pt_c_co, pt_f_mo, pt_f_co, ...
        FCGenes, states_c, states_f, targettop);
    title(t, sprintf('Numerically Best Configuration %d, KL = %f', ...
        topk, Y(b(topk))));
end

%%

% kx=0;
% for topk = 1:numComb
%     if ~isideal(topk), continue; end
%     kx=kx+1;
%     t = fun_drawreshisto(topk, Cc, configsK, ...
%         pt_c_mo, pt_c_co, pt_f_mo, pt_f_co, ...
%         FCGenes, states_c, states_f, targettop);
%     title(t, sprintf('Biologically Ideal Configuration, KL = %f', idealY(kx)));
% end

[idealY_sorted, idx] = sort(idealY);
a = find(isideal);
a = a(idx);
for topk = 1:3  % length(a)
    t = hlp.fun_drawreshisto(a(topk), Cc, configsK, ...
        pt_c_mo, pt_c_co, pt_f_mo, pt_f_co, ...
        FCGenes, states_c, states_f, targettop);
    title(t, sprintf('Biologically Ideal Configuration %d, KL = %f', ...
        topk, idealY_sorted(topk)));
end



