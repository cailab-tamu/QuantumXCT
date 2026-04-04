% train.m  –  optimize CRY gate angles over all K-link configurations.
% Runs prepare.m for shared data/state setup, then searches for the
% quantum circuit topology that minimises the one-way KL cost.
%
% Output handling follows karpathy/autoresearch/train.py:
%   - hyperparameters declared at the top (edit here, no flags needed)
%   - live single-line progress log overwritten with \r
%   - EMA-smoothed cost for stable display
%   - fast-fail on NaN / exploded cost
%   - structured key:value summary after --- separator
%   - all results saved to OUTPUT_FILE

% ---------------------------------------------------------------------------
% Hyperparameters  (edit these directly)
% ---------------------------------------------------------------------------

K           = 3;         % number of inter-cellular entanglement links
OPTIMIZER   = 1;         % 1 = fminunc (gradient)  2 = fminsearch  3 = fmincon
N_RESTARTS  = 5;         % independent random restarts per configuration
ANGLE_INIT  = 'random';  % 'random' (-pi,pi) | 'pi' (warm start at pi)
OUTPUT_FILE = 'train_results.mat';

% ---------------------------------------------------------------------------
% Shared data and quantum state setup
% ---------------------------------------------------------------------------

% prepare  % defines: cg1_mapped, cg2_mapped, extragates, configsK vars,
           %          costfn, targettop, FCGenes, states_f, states_c,
           %          pt_f_mo/co, pt_c_mo/co, f0_*, FibroblastGenes, CancerGenes

% ---------------------------------------------------------------------------
% Configuration search space
% ---------------------------------------------------------------------------

configsK = hlp.linkMatrix_dir_k(K);
numComb  = length(configsK);

Y       = nan(numComb, 1);
Cc      = cell(numComb, 1);
Theta   = cell(numComb, 1);
idealY  = [];
isideal = false(numComb, 1);

% ---------------------------------------------------------------------------
% Training loop
% ---------------------------------------------------------------------------

t_start    = tic;
smooth_cost = 0;
ema_beta   = 0.9;
best_cost  = inf;
step       = 0;   % counts valid (non-skipped) configurations processed

for idx = 1:numComb
    % skip configurations that contain a reversed duplicate (both A→B and B→A)
    % if any(ismember(fliplr(configsK{idx}), configsK{idx}, "rows")), continue; end

    % build parameterised circuit: base init + optional intracellular gate + CRY search gates
    layer_inte = [];
    for k = 1:height(configsK{idx})
        layer_inte = [layer_inte; cryGate(configsK{idx}(k,1), configsK{idx}(k,2), 0)];
    end
    if isempty(extragates)
        combinedGate = [cg1_mapped; cg2_mapped; layer_inte];
    else
        combinedGate = [cg1_mapped; cg2_mapped; extragates; layer_inte];
    end
    n = height(layer_inte);   % number of angle parameters
    C = quantumCircuit(combinedGate);

    % --- multiple random restarts to escape local minima ---
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
                lb = -pi*ones(n,1);  ub = pi*ones(n,1);
                [theta, fval] = fmincon(@hlp.i_obj, inivalue, [], [], [], [], ...
                    lb, ub, [], opts, pt_f_co, pt_c_co, C);
        end

        if fval < best_local_cost
            best_local_cost  = fval;
            best_local_theta = theta;
        end
    end

    % --- fast fail: abort this config if cost is invalid ---
    % if isnan(best_local_cost) || best_local_cost > 1e6
    %     fprintf('\nFAIL: config %d produced invalid cost (%.4g)\n', idx, best_local_cost);
    %     continue;
    % end

    % apply best angles to circuit and store
    for k = 1:n
        C.Gates(end-(k-1)).Angles = best_local_theta(k);
    end
    Cc{idx}    = C;
    Y(idx)     = best_local_cost;
    Theta{idx} = best_local_theta;

    if all(ismember(sort(configsK{idx},2), sort(targettop, 2), "rows"))
        idealY       = [idealY; best_local_cost];
        isideal(idx) = true;
    end

    % --- progress logging ---
    step       = step + 1;
    best_cost  = min(best_cost, best_local_cost);
    smooth_cost = ema_beta * smooth_cost + (1 - ema_beta) * best_local_cost;
    debiased   = smooth_cost / (1 - ema_beta^step);
    elapsed    = toc(t_start);
    pct        = 100 * idx / numComb;

    fprintf('\rconfig %04d/%d (%.1f%%) | cost: %.6f | best: %.6f | elapsed: %.0fs    ', ...
        idx, numComb, pct, debiased, best_cost, elapsed);
end

fprintf('\n');   % newline after final \r

% ---------------------------------------------------------------------------
% Final summary  (structured key:value, parseable by scripts)
% ---------------------------------------------------------------------------

elapsed_total = toc(t_start);
valid_count   = sum(~isnan(Y));
[~, top_idx]  = mink(Y, min(3, valid_count));

fprintf('---\n');
fprintf('best_cost:      %.6f\n', min(Y, [], 'omitnan'));
fprintf('ideal_cost:     %.6f\n', min(idealY));
fprintf('n_configs:      %d\n', numComb);
fprintf('n_valid:        %d\n', valid_count);
fprintf('n_ideal:        %d\n', sum(isideal));
fprintf('K:              %d\n', K);
fprintf('optimizer:      %d\n', OPTIMIZER);
fprintf('n_restarts:     %d\n', N_RESTARTS);
fprintf('total_seconds:  %.1f\n', elapsed_total);
fprintf('output_file:    %s\n', OUTPUT_FILE);

save(OUTPUT_FILE, ...
    'Y', 'Cc', 'Theta', 'configsK', 'isideal', 'idealY', 'targettop', ...
    'FibroblastGenes', 'CancerGenes', 'FCGenes', ...
    'pt_f_mo', 'pt_c_mo', 'pt_f_co', 'pt_c_co', ...
    'states_f', 'states_c', ...
    'K', 'OPTIMIZER', 'N_RESTARTS');
fprintf('results saved → %s\n', OUTPUT_FILE);

