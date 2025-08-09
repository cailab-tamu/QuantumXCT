classdef PCAlgorithm < handle
    % PCAlgorithm - Implementation of the PC (Peter-Clark) algorithm for causal discovery
    %
    % The PC algorithm learns causal structure through:
    % 1. Skeleton discovery using conditional independence tests
    % 2. Edge orientation using v-structures and propagation rules
    
    properties
        alpha                    % Significance level for independence tests
        max_k                   % Maximum conditioning set size
        variables               % Variable names
        n_vars                  % Number of variables
        data                    % Input data matrix
        skeleton                % Undirected graph (adjacency matrix)
        separating_sets         % Cell array storing separating sets
        directed_graph          % Final directed graph (adjacency matrix)
    end
    
    methods
        function obj = PCAlgorithm(alpha, max_conditioning_set_size)
            % Constructor
            % Inputs:
            %   alpha - Significance level (default: 0.05)
            %   max_conditioning_set_size - Max size of conditioning sets (default: 3)
            
            if nargin < 1 || isempty(alpha)
                obj.alpha = 0.05;
            else
                obj.alpha = alpha;
            end
            
            if nargin < 2 || isempty(max_conditioning_set_size)
                obj.max_k = 3;
            else
                obj.max_k = max_conditioning_set_size;
            end
        end
        
        function directed_graph = fit(obj, data, var_names)
            % Learn causal structure from data
            % Inputs:
            %   data - n x p matrix (n samples, p variables)
            %   var_names - cell array of variable names (optional)
            % Outputs:
            %   directed_graph - p x p adjacency matrix of learned DAG
            
            [n_samples, obj.n_vars] = size(data);
            obj.data = data;
            
            if nargin < 3 || isempty(var_names)
                obj.variables = cellstr(num2str((1:obj.n_vars)', 'X%d'));
            else
                obj.variables = var_names;
            end
            
            % Initialize separating sets
            obj.separating_sets = cell(obj.n_vars, obj.n_vars);
            
            % Phase 1: Learn skeleton
            fprintf('Phase 1: Learning skeleton...\n');
            obj.skeleton = obj.learn_skeleton();
            
            % Phase 2: Orient edges
            fprintf('Phase 2: Orienting edges...\n');
            obj.directed_graph = obj.orient_edges();
            
            directed_graph = obj.directed_graph;
        end
        
        function skeleton = learn_skeleton(obj)
            % Learn the skeleton (undirected graph) using conditional independence tests
            
            % Start with complete graph
            skeleton = ones(obj.n_vars) - eye(obj.n_vars);
            
            % Test conditional independence for increasing conditioning set sizes
            for k = 0:obj.max_k
                fprintf('  Testing with conditioning sets of size %d\n', k);
                
                % Find all edges to test
                [i_edges, j_edges] = find(triu(skeleton, 1));
                
                for edge_idx = 1:length(i_edges)
                    i = i_edges(edge_idx);
                    j = j_edges(edge_idx);
                    
                    if skeleton(i, j) == 0
                        continue; % Edge already removed
                    end
                    
                    % Get potential conditioning variables
                    neighbors_i = find(skeleton(i, :));
                    neighbors_j = find(skeleton(j, :));
                    neighbors_i(neighbors_i == j) = [];
                    neighbors_j(neighbors_j == i) = [];
                    potential_z = unique([neighbors_i, neighbors_j]);
                    
                    % Test all conditioning sets of size k
                    if length(potential_z) >= k
                        z_sets = nchoosek(potential_z, k);
                        if k == 0
                            z_sets = [];
                        end
                        
                        for z_idx = 1:size(z_sets, 1)
                            if k == 0
                                z_set = [];
                            else
                                z_set = z_sets(z_idx, :);
                            end
                            
                            if obj.test_conditional_independence(i, j, z_set)
                                % Remove edge and store separating set
                                skeleton(i, j) = 0;
                                skeleton(j, i) = 0;
                                obj.separating_sets{i, j} = z_set;
                                obj.separating_sets{j, i} = z_set;
                                break;
                            end
                        end
                    end
                end
            end
        end
        
        function is_independent = test_conditional_independence(obj, x, y, z)
            % Test conditional independence between variables x and y given z
            % Uses partial correlation for continuous variables
            
            if isempty(z)
                % Test marginal independence
                is_independent = obj.test_marginal_independence(x, y);
            else
                % Calculate partial correlation
                partial_corr = obj.partial_correlation(x, y, z);
                
                % Fisher's z-transformation for significance test
                n = size(obj.data, 1);
                z_score = 0.5 * log((1 + partial_corr) / (1 - partial_corr)) * sqrt(n - length(z) - 3);
                p_value = 2 * (1 - normcdf(abs(z_score)));
                
                is_independent = p_value > obj.alpha;
            end
        end
        
        function is_independent = test_marginal_independence(obj, x, y)
            % Test marginal independence between variables x and y
            
            [corr_coeff, p_value] = corrcoef(obj.data(:, x), obj.data(:, y));
            p_value = p_value(1, 2);
            is_independent = p_value > obj.alpha;
        end
        
        function partial_corr = partial_correlation(obj, x, y, z)
            % Calculate partial correlation between x and y given z
            
            if isempty(z)
                corr_matrix = corrcoef(obj.data(:, x), obj.data(:, y));
                partial_corr = corr_matrix(1, 2);
            else
                % Create subset of variables
                vars_subset = [x, y, z];
                data_subset = obj.data(:, vars_subset);
                
                % Calculate correlation matrix
                corr_matrix = corrcoef(data_subset);
                
                % Calculate partial correlation using matrix inversion
                try
                    inv_corr = inv(corr_matrix);
                    partial_corr = -inv_corr(1, 2) / sqrt(inv_corr(1, 1) * inv_corr(2, 2));
                catch
                    % Fallback if matrix is singular
                    corr_matrix = corrcoef(obj.data(:, x), obj.data(:, y));
                    partial_corr = corr_matrix(1, 2);
                end
            end
        end
        
        function directed_graph = orient_edges(obj)
            % Orient edges in the skeleton to form a directed acyclic graph
            
            % Convert skeleton to directed graph (both directions for each edge)
            directed_graph = obj.skeleton;
            
            % Rule 1: Orient v-structures (colliders)
            obj.orient_v_structures(directed_graph);
            
            % Rules 2-4: Apply Meek's orientation rules
            changed = true;
            while changed
                changed = false;
                changed = obj.meek_rule_1(directed_graph) || changed;
                changed = obj.meek_rule_2(directed_graph) || changed;
                changed = obj.meek_rule_3(directed_graph) || changed;
            end
        end
        
        function orient_v_structures(obj, graph)
            % Identify and orient v-structures (X -> Z <- Y where X and Y are not adjacent)
            
            for z = 1:obj.n_vars
                neighbors = find(obj.skeleton(z, :));
                
                % Check all pairs of non-adjacent neighbors
                for i = 1:length(neighbors)
                    for j = i+1:length(neighbors)
                        x = neighbors(i);
                        y = neighbors(j);
                        
                        % Check if x and y are not adjacent in skeleton
                        if obj.skeleton(x, y) == 0
                            % Check if z is not in the separating set of x and y
                            sep_set = obj.separating_sets{x, y};
                            if isempty(sep_set) || ~ismember(z, sep_set)
                                % Orient as v-structure: x -> z <- y
                                graph(z, x) = 0;
                                graph(z, y) = 0;
                            end
                        end
                    end
                end
            end
        end
        
        function changed = meek_rule_1(obj, graph)
            % Rule 1: If X -> Y - Z and X and Z not adjacent, then orient Y -> Z
            changed = false;
            
            for y = 1:obj.n_vars
                % Find directed edges into y (parents)
                parents_y = find(graph(:, y)' & ~graph(y, :));
                % Find undirected edges from y
                undirected_from_y = find(graph(y, :) & graph(:, y)');
                
                for x = parents_y
                    for z = undirected_from_y
                        % Check if x and z are not adjacent
                        if obj.skeleton(x, z) == 0
                            % Orient y -> z
                            graph(z, y) = 0;
                            changed = true;
                        end
                    end
                end
            end
        end
        
        function changed = meek_rule_2(obj, graph)
            % Rule 2: If X -> Y -> Z and X - Z, then orient X -> Z
            changed = false;
            
            for y = 1:obj.n_vars
                parents_y = find(graph(:, y)' & ~graph(y, :));
                children_y = find(graph(y, :) & ~graph(:, y)');
                
                for x = parents_y
                    for z = children_y
                        % Check if x - z (undirected)
                        if graph(x, z) && graph(z, x)
                            % Orient x -> z
                            graph(z, x) = 0;
                            changed = true;
                        end
                    end
                end
            end
        end
        
        function changed = meek_rule_3(obj, graph)
            % Rule 3: If X - Y, X - Z, Y -> W, Z -> W, and Y and Z not adjacent, then X -> Y
            changed = false;
            
            for x = 1:obj.n_vars
                % Find undirected neighbors of x
                undirected_neighbors = find(graph(x, :) & graph(:, x)');
                
                if length(undirected_neighbors) >= 2
                    neighbor_pairs = nchoosek(undirected_neighbors, 2);
                    
                    for pair_idx = 1:size(neighbor_pairs, 1)
                        y = neighbor_pairs(pair_idx, 1);
                        z = neighbor_pairs(pair_idx, 2);
                        
                        % Check if y and z are not adjacent
                        if obj.skeleton(y, z) == 0
                            % Find common children of y and z
                            children_y = find(graph(y, :) & ~graph(:, y)');
                            children_z = find(graph(z, :) & ~graph(:, z)');
                            common_children = intersect(children_y, children_z);
                            
                            if ~isempty(common_children)
                                % Orient x -> y and x -> z
                                graph(y, x) = 0;
                                graph(z, x) = 0;
                                changed = true;
                            end
                        end
                    end
                end
            end
        end
        
        function plot_results(obj)
            % Plot both skeleton and final directed graph
            
            figure('Position', [100, 100, 1200, 500]);
            
            % Plot skeleton
            subplot(1, 2, 1);
            obj.plot_graph(obj.skeleton, 'Learned Skeleton', false);
            
            % Plot directed graph  
            subplot(1, 2, 2);
            obj.plot_graph(obj.directed_graph, 'Final Causal Graph', true);
        end
        
        function plot_graph(obj, adj_matrix, title_str, is_directed)
            % Plot a graph given its adjacency matrix
            
            % Create layout using spring-embedder algorithm
            pos = obj.spring_layout(adj_matrix);
            
            % Plot edges
            hold on;
            for i = 1:obj.n_vars
                for j = 1:obj.n_vars
                    if adj_matrix(i, j) == 1
                        if is_directed && adj_matrix(j, i) == 0
                            % Directed edge
                            obj.draw_arrow(pos(i, :), pos(j, :), 'r');
                        elseif ~is_directed || adj_matrix(j, i) == 1
                            % Undirected edge (only draw once)
                            if i < j
                                plot([pos(i, 1), pos(j, 1)], [pos(i, 2), pos(j, 2)], 'b-', 'LineWidth', 1.5);
                            end
                        end
                    end
                end
            end
            
            % Plot nodes
            scatter(pos(:, 1), pos(:, 2), 500, 'filled', 'MarkerFaceColor', [0.7, 0.7, 0.9]);
            
            % Add labels
            for i = 1:obj.n_vars
                text(pos(i, 1), pos(i, 2), obj.variables{i}, ...
                     'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
            end
            
            title(title_str, 'FontSize', 14);
            axis equal;
            axis off;
            hold off;
        end
        
        function pos = spring_layout(obj, adj_matrix)
            % Simple spring-embedder layout algorithm
            
            n = obj.n_vars;
            pos = rand(n, 2) * 10; % Random initial positions
            
            for iter = 1:100
                forces = zeros(n, 2);
                
                % Repulsive forces between all nodes
                for i = 1:n
                    for j = 1:n
                        if i ~= j
                            diff = pos(i, :) - pos(j, :);
                            dist = norm(diff);
                            if dist > 0
                                forces(i, :) = forces(i, :) + diff / dist^2;
                            end
                        end
                    end
                end
                
                % Attractive forces for connected nodes
                for i = 1:n
                    for j = 1:n
                        if adj_matrix(i, j) == 1 || adj_matrix(j, i) == 1
                            diff = pos(j, :) - pos(i, :);
                            dist = norm(diff);
                            if dist > 0
                                forces(i, :) = forces(i, :) + diff * dist * 0.1;
                            end
                        end
                    end
                end
                
                % Update positions
                pos = pos + forces * 0.01;
            end
        end
        
        function draw_arrow(obj, start_pos, end_pos, color)
            % Draw an arrow from start_pos to end_pos
            
            % Calculate arrow direction
            diff = end_pos - start_pos;
            len = norm(diff);
            unit_vec = diff / len;
            
            % Shorten arrow to not overlap with nodes
            arrow_start = start_pos + unit_vec * 0.5;
            arrow_end = end_pos - unit_vec * 0.5;
            
            % Draw main line
            plot([arrow_start(1), arrow_end(1)], [arrow_start(2), arrow_end(2)], ...
                 'Color', color, 'LineWidth', 2);
            
            % Draw arrowhead
            perp_vec = [-unit_vec(2), unit_vec(1)];
            head_size = 0.3;
            arrow_head1 = arrow_end - unit_vec * head_size + perp_vec * head_size * 0.5;
            arrow_head2 = arrow_end - unit_vec * head_size - perp_vec * head_size * 0.5;
            
            plot([arrow_end(1), arrow_head1(1)], [arrow_end(2), arrow_head1(2)], ...
                 'Color', color, 'LineWidth', 2);
            plot([arrow_end(1), arrow_head2(1)], [arrow_end(2), arrow_head2(2)], ...
                 'Color', color, 'LineWidth', 2);
        end
        
        function print_results(obj)
            % Print the results in a readable format
            
            fprintf('\nLearned Causal Structure:\n');
            fprintf('========================\n');
            
            edge_count = 0;
            for i = 1:obj.n_vars
                for j = 1:obj.n_vars
                    if obj.directed_graph(i, j) == 1 && obj.directed_graph(j, i) == 0
                        fprintf('%s -> %s\n', obj.variables{i}, obj.variables{j});
                        edge_count = edge_count + 1;
                    end
                end
            end
            
            fprintf('\nTotal directed edges: %d\n', edge_count);
            
            % Print adjacency matrix
            fprintf('\nAdjacency Matrix:\n');
            fprintf('================\n');
            fprintf('     ');
            for j = 1:obj.n_vars
                fprintf('%6s', obj.variables{j});
            end
            fprintf('\n');
            
            for i = 1:obj.n_vars
                fprintf('%4s ', obj.variables{i});
                for j = 1:obj.n_vars
                    fprintf('%6d', obj.directed_graph(i, j));
                end
                fprintf('\n');
            end
        end
    end
end

