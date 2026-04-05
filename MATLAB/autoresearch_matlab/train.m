% train.m  –  autoresearch: one ~4-min experiment run.
%
% Each run randomly samples K-link configurations within TIME_BUDGET,
% optimises gate angles, and reports best_cost.
%
% Workflow:
%   1. Modify hyperparameters below to test a hypothesis
%   2. Run:  diary run.log; run('train.m'); diary off
%   3. Compare best_cost to baseline — keep commit or git reset --hard HEAD~1

% ---------------------------------------------------------------------------
% Hyperparameters  (edit these directly)
% ---------------------------------------------------------------------------

K              = 5;          % number of inter-cellular entanglement links
TIME_BUDGET    = 240;        % seconds per run (~4 min)
OPTIMIZER      = 1;          % 1 = fminunc  2 = fminsearch  3 = fmincon
N_RESTARTS     = 1;          % random restarts per configuration
ANGLE_INIT     = 'random';   % 'random' (-pi,pi) | 'pi' (warm start)
GATE_TYPE      = 'cry';      % 'cry' | 'crx' | 'cx'
SKIP_BIDIRECT  = false;      % skip configs containing both A→B and B→A
SKIP_NONUNIQUE = false;      % skip configs whose qubits don't cover both bags
OUTPUT_FILE    = 'train_results.mat';

% ---------------------------------------------------------------------------
% Shared data and quantum state setup
% ---------------------------------------------------------------------------

% prepare  % defines: cg1_mapped, cg2_mapped, extragates,
           %          FibroblastGenes, CancerGenes, FCGenes,
           %          states_f, states_c, pt_f_mo/co, pt_c_mo/co

% ---------------------------------------------------------------------------
% Configuration space — random sample within time budget
% ---------------------------------------------------------------------------

n1 = round(log2(length(pt_f_mo)));
n2 = round(log2(length(pt_c_mo)));

configsK = hlp.linkMatrix_dir_k(K);
numComb  = length(configsK);

% shuffle all configs for random exploration each run
order = randperm(numComb);

Y     = nan(numComb, 1);
Cc    = cell(numComb, 1);
Theta = cell(numComb, 1);

t_start     = tic;
smooth_cost = 0;
ema_beta    = 0.9;
best_cost   = inf;
step        = 0;

for ui = 1:numComb
    if toc(t_start) >= TIME_BUDGET, break; end

    idx = order(ui);

    % --- skip filters ---
    if SKIP_BIDIRECT  && any(ismember(fliplr(configsK{idx}), configsK{idx}, "rows")), continue; end
    if SKIP_NONUNIQUE && numel(unique(configsK{idx}(:))) < (n1 + n2),                 continue; end

    % --- build parameterised circuit ---
    layer_inte = [];
    for k = 1:height(configsK{idx})
        src = configsK{idx}(k, 1);
        tgt = configsK{idx}(k, 2);
        switch GATE_TYPE
            case 'cry';  layer_inte = [layer_inte; cryGate(src, tgt, 0)];
            case 'crx';  layer_inte = [layer_inte; crxGate(src, tgt, 0)];
            case 'cx';   layer_inte = [layer_inte; cxGate(src, tgt)];
        end
    end
    if isempty(extragates)
        combinedGate = [cg1_mapped; cg2_mapped; layer_inte];
    else
        combinedGate = [cg1_mapped; cg2_mapped; extragates; layer_inte];
    end
    n = height(layer_inte);
    C = quantumCircuit(combinedGate);

    % --- optimize angles (or direct evaluation for CX) ---
    if strcmp(GATE_TYPE, 'cx')
        best_local_cost  = hlp.i_obj(zeros(n, 1), pt_f_co, pt_c_co, C);
        best_local_theta = zeros(n, 1);
    else
        best_local_cost  = inf;
        best_local_theta = zeros(n, 1);

        for rep = 1:N_RESTARTS
            switch ANGLE_INIT
                case 'random';  inivalue = -pi + 2*pi*rand(n, 1);
                case 'pi';      inivalue =  pi*ones(n, 1);
            end

            switch OPTIMIZER
                case 1
                    opts = optimset('Display', 'none');
                    [theta, fval] = fminunc(@hlp.i_obj, inivalue, opts, pt_f_co, pt_c_co, C);
                case 2
                    opts = optimset('Display', 'none');
                    [theta, fval] = fminsearch(@hlp.i_obj, inivalue, opts, pt_f_co, pt_c_co, C);
                case 3
                    opts = optimoptions('fmincon', 'Display', 'none');
                    lb = -pi*ones(n, 1);  ub = pi*ones(n, 1);
                    [theta, fval] = fmincon(@hlp.i_obj, inivalue, [], [], [], [], ...
                        lb, ub, [], opts, pt_f_co, pt_c_co, C);
            end

            if fval < best_local_cost
                best_local_cost  = fval;
                best_local_theta = theta;
            end
        end

        for k = 1:n
            C.Gates(end-(k-1)).Angles = best_local_theta(k);
        end
    end

    Cc{idx}    = C;
    Y(idx)     = best_local_cost;
    Theta{idx} = best_local_theta;

    % --- progress ---
    step        = step + 1;
    best_cost   = min(best_cost, best_local_cost);
    smooth_cost = ema_beta * smooth_cost + (1 - ema_beta) * best_local_cost;
    debiased    = smooth_cost / (1 - ema_beta^step);
    elapsed     = toc(t_start);

    fprintf('\r[%3.0fs] %d sampled | cost: %.6f | best: %.6f    ', ...
        elapsed, step, debiased, best_cost);
end

fprintf('\n');

% ---------------------------------------------------------------------------
% Summary
% ---------------------------------------------------------------------------

elapsed_total = toc(t_start);
valid         = ~isnan(Y);

fprintf('---\n');
fprintf('best_cost:      %.6f\n', min(Y, [], 'omitnan'));
fprintf('n_sampled:      %d/%d\n', step, numComb);
fprintf('K:              %d\n',   K);
fprintf('optimizer:      %d\n',   OPTIMIZER);
fprintf('n_restarts:     %d\n',   N_RESTARTS);
fprintf('gate_type:      %s\n',   GATE_TYPE);
fprintf('total_seconds:  %.1f\n', elapsed_total);
fprintf('output_file:    %s\n',   OUTPUT_FILE);

save(OUTPUT_FILE, ...
    'Y', 'Cc', 'Theta', 'configsK', ...
    'FibroblastGenes', 'CancerGenes', 'FCGenes', ...
    'pt_f_mo', 'pt_c_mo', 'pt_f_co', 'pt_c_co', ...
    'states_f', 'states_c', ...
    'K', 'OPTIMIZER', 'N_RESTARTS', 'GATE_TYPE');
fprintf('results saved → %s\n', OUTPUT_FILE);
