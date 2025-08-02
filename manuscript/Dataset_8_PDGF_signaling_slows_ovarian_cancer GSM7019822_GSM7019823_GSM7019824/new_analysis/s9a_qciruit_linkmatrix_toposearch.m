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
[pt_f_co, ~, ~, ~] = getn(batquery, genquery, sce, false);

batquery = "Cancer Cells (Co)";
genquery = CancerGenes;
[pt_c_co, ~, ~, ~] = getn(batquery, genquery, sce, false);


%% 
% quantum state preparation via uniformly controlled rotations
amp = sqrt(pt_f_mo);                % unnormalized amplitudes vector
cg_f = initGate(1:3, amp);          % creates CompositeGate to initialize qubits 1–3 :contentReference[oaicite:1]{index=1}
qc_f = quantumCircuit(cg_f);        % initialize circuit with the gate

amp = sqrt(pt_c_mo);               % unnormalized amplitudes vector
cg_c = initGate(1:4, amp);         % creates CompositeGate to initialize qubits 1–3 :contentReference[oaicite:1]{index=1}
qc_c = quantumCircuit(cg_c);         % initialize circuit with the gate


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

% Number of items in Bag 1 and Bag 2
n1 = 3;
n2 = 4;

numLinks = n1 * n2;
numComb = 2^numLinks;

bag1_labels = {'A','B','C'};
bag2_labels = {'1','2','3','4'};

bag1_labels = {1, 2, 3};
bag2_labels = {4, 5, 6, 7};

Y = nan(numComb, 1);
Cc = cell(numComb, 1);

for idx = 0:numComb-1
    % Get binary vector for current combination
    bits = bitget(idx, 1:numLinks);
    % Reshape to 3x4 "link matrix"
    linkmat = reshape(bits, n2, n1)';
    
    [r,c] = find(linkmat);
    
    if isempty(r)
        % disp('  No links.');
    else
        layer_inte = [];
        for k = 1:numel(r)
            % fprintf('  Link: %s - %s\n', bag1_labels{r(k)}, bag2_labels{c(k)});
            layer_inte = [layer_inte; cxGate(bag2_labels{c(k)}, bag1_labels{r(k)})];
        end
        combinedGate = [cg1_mapped; cg2_mapped; layer_inte];
        C = quantumCircuit(combinedGate);
        Cc{idx+1} = C;
        S = simulate(C);
        [~, po_f] = querystates(S,[1 2 3]);     % observed state pattern in fibroblast
        [~, po_c] = querystates(S,[4 5 6 7]);   % observed state pattern in cancer   
        kl1 = i_kldiverg(pt_f_co, po_f);
        kl2 = i_kldiverg(pt_c_co, po_c);
        Y(idx+1) = kl1 + kl2;        
    end
    % Print the links for this combination
    fprintf('Combination %d: %f\n', idx+1, Y(idx+1));
    % fprintf('\n');
end

[a,b]=mink(Y,5)



C = Cc{b(1)};
S = simulate(C);
% [~, po_f] = querystates(S,[1 2 3]);     % observed state pattern in fibroblast
% [~, po_c] = querystates(S,[4 5 6 7]);   % observed state pattern in cancer   
% kl1 = i_kldiverg(pt_f_co, po_f);
% kl2 = i_kldiverg(pt_c_co, po_c);
% Y(idx+1) = kl1 + kl2;        





% layer_inte = [cryGate(5,4,0); ...
%         cryGate(1,6,0); ...
%         cryGate(7,2,0); ...
%         cryGate(3,5,0)];





%{
n = size(layer_inte, 1);
inivalue = -pi + (2*pi)*rand(n, 1);

methodid = 3;
switch methodid
    case 1
        options = optimset('Display','iter');
        [optimtheta, fval] = fminunc(@i_obj, inivalue, options, pt_f_co, pt_c_co, C);       
    case 2
        options = optimset('Display','iter');
        [optimtheta, fval] = fminsearch(@i_obj, inivalue, options, pt_f_co, pt_c_co, C);
    case 3
        options = optimoptions('fmincon','Display','iter');
        lb = -pi*ones(n, 1); ub = pi*ones(n, 1);
        [optimtheta, fval] = fmincon(@i_obj, inivalue, [], [], [], [], ...
            lb, ub, [], options, pt_f_co, pt_c_co, C);
end


S = simulate(C);    
[states_f, po_f_mo] = querystates(S,[1 2 3]);     % observed state pattern in fibroblast
[states_c, po_c_mo] = querystates(S,[4 5 6 7]);   % observed state pattern in cancer   

for k = 1:n
    C.Gates(end-(k-1)).Angles = optimtheta(k);
end
%}

S = simulate(C);
[states_f2, po_f2] = querystates(S,[1 2 3]);     % observed state pattern in fibroblast
[states_c2, po_c2] = querystates(S,[4 5 6 7]);   % observed state pattern in cancer   


figure;
nexttile
plot(C)

nexttile
bar([pt_c_mo pt_c_co po_c2])
set(gca,'XTick',1:length(states_c));
set(gca,'XTickLabel',states_c);
ylabel('Freq. of cells');
xlabel('Expression pattern');
title('Cancer Cells')
legend({'Mono','Co','Simulated2'})

nexttile
bar([pt_f_mo pt_f_co po_f2])
set(gca,'XTick',1:length(states_f));
set(gca,'XTickLabel',states_f);
ylabel('Freq. of cells');
xlabel('Expression pattern');
title('Fibroblasts')
legend({'Mono','Co','Simulated2'})

