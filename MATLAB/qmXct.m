function qmXct(sce, CellType1Genes, CellType2Genes, tags, targettoplinkers, K, extragates)

    if nargin < 6, K = 3; end
    if nargin < 7, extragates = []; end
    
    % targettoplinkers = [7 2; 1 6; 3 5];
    % CellType1Genes = ["TGFB1", "PDGFRB", "IL6"];
    % CellType2Genes = ["STAT3","IL6RorST","TGFBR1or2","PDGFB"];
    % tags = ["Fibroblasts (Mo)", "Cancer Cells (Mo)", "Fibroblasts (Co)", "Cancer Cells (Co)"];
    % extragates = cxGate(4,5);
    
    T1T2Genes = [CellType1Genes CellType2Genes];
    assert(all(ismember(T1T2Genes, sce.g)))
    
    %    "Co_GSM8882016Cardiomyocytes"
    %    "Co_GSM8882016Epicardial cells"
    %    "Mo_GSM8882017Cardiomyocytes"
    %    "Mo_GSM8882017Epicardial cells"
    
    [pt_f_mo, ~, ~, ~] = hlp.getn(tags(1), CellType1Genes, sce, false);
    [pt_c_mo, ~, ~, ~] = hlp.getn(tags(2), CellType2Genes, sce, false);
    [pt_f_co, ~, ~, ~] = hlp.getn(tags(3), CellType1Genes, sce, false);
    [pt_c_co, ~, ~, ~] = hlp.getn(tags(4), CellType2Genes, sce, false);
     
    n1 = log2(numel(pt_f_mo));
    cg_1 = initGate(1:n1, sqrt(pt_f_mo));   % creates CompositeGate to initialize qubits 1–3 :contentReference[oaicite:1]{index=1}
    n2 = log2(numel(pt_c_mo));
    cg_2 = initGate(1:n2, sqrt(pt_c_mo));   % creates CompositeGate to initialize qubits 1–4 :contentReference[oaicite:1]{index=1}
    
    %%
    cg1_mapped = compositeGate(cg_1, 1:n1);
    cg2_mapped = compositeGate(cg_2, n1+1:n1+n2);
    combinedGate = [cg1_mapped; cg2_mapped];
    
    C = quantumCircuit(combinedGate);
    S = simulate(C);
    [states_f, po_f_mo] = querystates(S,1:n1);
    [states_c, po_c_mo] = querystates(S,n1+1:n1+n2);
    
    %%
    bag1 = num2cell(1:n1);
    bag2 = num2cell(n1+1:n1+n2);
    configsK = hlp.linkMatrix_dir_k(K, bag1, bag2);
    numComb = length(configsK);
    
    Y = nan(numComb, 1);
    Cc = cell(numComb, 1);
    idealY = [];
    isideal = false(numComb, 1);
    
    for idx = 1:numComb
        if any(ismember(fliplr(configsK{idx}), configsK{idx}, "rows")), continue; end
        % if length(unique(configsK{idx}))<6, continue; end
        layer_inte = [];
        for k = 1:height(configsK{idx})
            layer_inte = [layer_inte; cxGate(configsK{idx}(k,1), configsK{idx}(k,2))];
        end
    
        if ~isempty(extragates)
            combinedGate = [cg1_mapped; cg2_mapped; extragates; layer_inte];
        else
            combinedGate = [cg1_mapped; cg2_mapped; layer_inte];
        end
    
        C = quantumCircuit(combinedGate);
        Cc{idx} = C;
        S = simulate(C);
        [~, po_f] = querystates(S,1:n1);         % observed state pattern in fibroblast
        [~, po_c] = querystates(S,n1+1:n1+n2);   % observed state pattern in cancer   
        kl1 = i_kldiverg(pt_f_co, po_f);
        kl2 = i_kldiverg(pt_c_co, po_c);
        Y(idx) = kl1 + kl2;
        fprintf('Combination %d: %f\n', idx, Y(idx));
        
        if all(ismember(sort(configsK{idx},2), sort(targettoplinkers, 2),"rows"))
            idealY = [idealY; Y(idx)];
            isideal(idx) = true;
        end
    end
    
    %%
    [~, b] = mink(Y, 3);
    
    figure;
    plot(Y);
    yline(idealY,'r');
    for topk = 1:3
        t = hlp.fun_drawreshisto(b(topk), Cc, configsK, ...
            pt_c_mo, pt_c_co, pt_f_mo, pt_f_co, ...
            T1T2Genes, states_c, states_f, targettoplinkers);
        title(t, sprintf('Numerically Best Configuration %d, KL = %f', ...
            topk, Y(b(topk))));
    end
    
    [idealY_sorted, idx] = sort(idealY);
    a = find(isideal);
    a = a(idx);
    for topk = 1:min([10 length(a)])
        t = hlp.fun_drawreshisto(a(topk), Cc, configsK, ...
            pt_c_mo, pt_c_co, pt_f_mo, pt_f_co, ...
            T1T2Genes, states_c, states_f, targettoplinkers);
        title(t, sprintf('Biologically Ideal Configuration %d, KL = %f', ...
            topk, idealY_sorted(topk)));
    end

end