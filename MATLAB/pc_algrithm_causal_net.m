%% Example usage and testing functions

function data = generate_sample_data(n_samples)
    % Generate sample data with known causal structure: A -> B -> C, A -> D -> C
    
    if nargin < 1
        n_samples = 1000;
    end
    
    rng(42); % For reproducibility
    
    % Generate data according to: A -> B -> C, A -> D -> C
    A = randn(n_samples, 1);
    B = 0.8 * A + 0.5 * randn(n_samples, 1);
    D = 0.6 * A + 0.5 * randn(n_samples, 1);
    C = 0.7 * B + 0.5 * D + 0.3 * randn(n_samples, 1);
    
    data = [A, B, C, D];
end

function run_pc_example()
    % Example of how to use the PC algorithm
    
    fprintf('PC Algorithm Example\n');
    fprintf('===================\n');
    
    % Generate sample data
    fprintf('Generating sample data with structure: A -> B -> C, A -> D -> C\n');
    data = generate_sample_data(1000);
    var_names = {'A', 'B', 'C', 'D'};
    
    % Run PC algorithm
    pc = PCAlgorithm(0.05, 2);
    learned_graph = pc.fit(data, var_names);
    
    % Print and plot results
    pc.print_results();
    pc.plot_results();
end

% Run the example
run_pc_example();
