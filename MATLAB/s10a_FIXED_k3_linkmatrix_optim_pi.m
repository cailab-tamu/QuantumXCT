% improved by adding intracellular link between ["STAT3","IL6RorST"] see
% line 63
a = pwd;
targettop = [7 2; 1 6; 3 5];
cd('../manuscript/Dataset_8_PDGF_signaling_slows_ovarian_cancer GSM7019822_GSM7019823_GSM7019824/new_analysis/');
run('s0_merge_subunit_genes.m')
% addpath ./
% addpath ../
cd(a);
%%
FibroblastGenes = ["TGFB1", "PDGFRB", "IL6"];
CancerGenes = ["STAT3","IL6RorST","TGFBR1or2","PDGFB"];
FCGenes = [FibroblastGenes CancerGenes];


batquery = "Fibroblasts (Mo)"; genquery = FibroblastGenes;
[pt_f_mo, ~, ~, f0_f_mo] = hlp.getn(batquery, genquery, sce, false);

batquery = "Cancer Cells (Mo)"; genquery = CancerGenes;
[pt_c_mo, ~, ~, f0_c_mo] = hlp.getn(batquery, genquery, sce, false);

batquery = "Fibroblasts (Co)"; genquery = FibroblastGenes;
[pt_f_co, ~, ~, ~] = hlp.getn(batquery, genquery, sce, false);

batquery = "Cancer Cells (Co)"; genquery = CancerGenes;
[pt_c_co, ~, ~, ~] = hlp.getn(batquery, genquery, sce, false);

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
configsK = hlp.linkMatrix_dir_k(3);
numComb = length(configsK);

Y = nan(numComb, 1);
Cc = cell(numComb, 1);
idealY = [];
isideal = false(numComb, 1);

for idx = 1:numComb

    if any(ismember(fliplr(configsK{idx}), configsK{idx}, "rows")), continue; end


    % layer_inte1 = [];
    % for k = 1:height(configsK{idx})
    %     layer_inte1 = [layer_inte1; cxGate(configsK{idx}(k,1), configsK{idx}(k,2))];
    % end
    layer_inte1 = [cxGate(7,2); cxGate(1,6); cxGate(3,5)];

    layer_inte = [];
    for k = 1:height(configsK{idx})
        layer_inte = [layer_inte; crxGate(configsK{idx}(k,1), configsK{idx}(k,2), 0)];
    end
    % combinedGate = [cg1_mapped; cg2_mapped; cxGate(5, 4); layer_inte1; layer_inte];
    % combinedGate = [cg1_mapped; cg2_mapped; cxGate(4, 5); layer_inte1; layer_inte];
    combinedGate = [cg1_mapped; cg2_mapped; cxGate(4, 5); layer_inte];


    n = size(layer_inte, 1);             % number of gates need parameters to be estimated
    % inivalue = -pi + (2*pi)*rand(n, 1);
    inivalue = pi*ones(n, 1);

    C = quantumCircuit(combinedGate);

    methodid = 1;
    switch methodid
        case 1
            options = optimset('Display','none');
            [optimtheta, fval] = fminunc(@hlp.i_obj, inivalue, options, pt_f_co, pt_c_co, C);       
        case 2
            options = optimset('Display','none');
            [optimtheta, fval] = fminsearch(@hlp.i_obj, inivalue, options, pt_f_co, pt_c_co, C);
        case 3
            options = optimoptions('fmincon','Display','none');
            lb = -pi*ones(n, 1); ub = pi*ones(n, 1);
            [optimtheta, fval] = fmincon(@hlp.i_obj, inivalue, [], [], [], [], ...
                lb, ub, [], options, pt_f_co, pt_c_co, C);
    end

    for k = 1:n
        C.Gates(end-(k-1)).Angles = optimtheta(k);
    end
    
    Cc{idx} = C;

    S = simulate(C);
    [states_f2, po_f] = querystates(S,[1 2 3]);     % observed state pattern in fibroblast
    [states_c2, po_c] = querystates(S,[4 5 6 7]);   % observed state pattern in cancer       
    
    kl1 = hlp.i_kldiverg(pt_f_co, po_f, true);
    kl2 = hlp.i_kldiverg(pt_c_co, po_c, true);
    Y(idx) = kl1 + kl2;
    fprintf('Combination %d: %f\n', idx, Y(idx));
    if all(ismember(sort(configsK{idx},2), sort(targettop, 2),"rows"))
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
    subtitle(t, sprintf('%.2f   ', optimtheta))

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
for topk = 1:length(a)
    t = hlp.fun_drawreshisto(a(topk), Cc, configsK, ...
        pt_c_mo, pt_c_co, pt_f_mo, pt_f_co, ...
        FCGenes, states_c, states_f, targettop);
    title(t, sprintf('Biologically Ideal Configuration %d, KL = %f', ...
        topk, idealY_sorted(topk)));
    subtitle(t, sprintf('%.2f   ', optimtheta))
end



