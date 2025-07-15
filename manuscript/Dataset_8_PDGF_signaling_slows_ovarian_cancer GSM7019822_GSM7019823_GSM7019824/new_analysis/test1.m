function quantum_state_preparation()
    % Quantum State Preparation in MATLAB
    % Creates a 3-qubit quantum circuit to match given frequency distribution
    
    % Your target frequencies
    target_frequencies = [0.2346, 0.1829, 0.0911, 0.0617, 0.1646, 0.1259, 0.0807, 0.0585];
    
    fprintf('Quantum State Preparation for 3 Qubits\n');
    fprintf('=====================================\n');
    
    % Method 1: Direct amplitude encoding
    fprintf('\nMethod 1: Direct Amplitude Encoding\n');
    [circuit1, state1] = create_amplitude_encoding_circuit(target_frequencies);
    measured_freq1 = simulate_measurements(state1, 10000);
    
    % Method 2: Systematic controlled rotations
    fprintf('\nMethod 2: Systematic Controlled Rotations\n');
    [circuit2, state2] = create_systematic_circuit(target_frequencies);
    measured_freq2 = simulate_measurements(state2, 10000);
    
    % Compare results
    fprintf('\nComparison of Results:\n');
    fprintf('State | Target   | Method1  | Method2  | Error1   | Error2\n');
    fprintf('------|----------|----------|----------|----------|----------\n');
    
    total_error1 = 0;
    total_error2 = 0;
    
    for i = 1:8
        binary = dec2bin(i-1, 3);
        error1 = abs(target_frequencies(i) - measured_freq1(i));
        error2 = abs(target_frequencies(i) - measured_freq2(i));
        total_error1 = total_error1 + error1;
        total_error2 = total_error2 + error2;
        
        fprintf('|%s⟩  | %.4f   | %.4f   | %.4f   | %.4f   | %.4f\n', ...
            binary, target_frequencies(i), measured_freq1(i), measured_freq2(i), error1, error2);
    end
    
    fprintf('\nTotal Absolute Errors:\n');
    fprintf('Method 1: %.4f\n', total_error1);
    fprintf('Method 2: %.4f\n', total_error2);
    
    % Visualize results
    visualize_results(target_frequencies, measured_freq1, measured_freq2);
end

function [circuit, final_state] = create_amplitude_encoding_circuit(frequencies)
    % Direct amplitude encoding method
    % Converts frequencies directly to quantum amplitudes
    
    % Normalize to probabilities
    probabilities = frequencies / sum(frequencies);
    
    % Convert to amplitudes (assuming real amplitudes)
    amplitudes = sqrt(probabilities);
    
    % Create initial state |000⟩
    initial_state = [1; 0; 0; 0; 0; 0; 0; 0];
    
    % The target state is directly the amplitude vector
    final_state = amplitudes';
    
    % Circuit description (conceptual)
    circuit = struct();
    circuit.method = 'Direct Amplitude Encoding';
    circuit.description = 'Directly sets amplitudes to sqrt(probabilities)';
    circuit.gates = {'State Preparation'};
    
    fprintf('Target amplitudes: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]\n', ...
        amplitudes);
end

function [circuit, final_state] = create_systematic_circuit(frequencies)
    % Systematic approach using controlled rotations
    % Builds the state hierarchically
    
    % Normalize to probabilities
    probabilities = frequencies / sum(frequencies);
    
    % Start with |000⟩ state
    state = [1; 0; 0; 0; 0; 0; 0; 0];
    
    % Circuit structure
    circuit = struct();
    circuit.method = 'Systematic Controlled Rotations';
    circuit.gates = {};
    gate_count = 0;
    
    % Level 1: Split between first 4 states (000-011) and last 4 states (100-111)
    prob_first_half = sum(probabilities(1:4));
    prob_second_half = sum(probabilities(5:8));
    
    if prob_first_half + prob_second_half > 0
        theta_0 = 2 * asin(sqrt(prob_second_half));
        state = apply_ry_gate(state, 1, theta_0);  % qubit 0 (1st qubit)
        gate_count = gate_count + 1;
        circuit.gates{gate_count} = sprintf('RY(%.4f) on qubit 0', theta_0);
    end
    
    % Level 2: Split within each half
    % First half: 000-001 vs 010-011
    if prob_first_half > 0
        prob_00x = sum(probabilities(1:2));
        prob_01x = sum(probabilities(3:4));
        if prob_00x + prob_01x > 0
            theta_1 = 2 * asin(sqrt(prob_01x / (prob_00x + prob_01x)));
            state = apply_controlled_ry_gate(state, 1, 2, theta_1);  % control: qubit 0, target: qubit 1
            gate_count = gate_count + 1;
            circuit.gates{gate_count} = sprintf('CRY(%.4f) control=0, target=1', theta_1);
        end
    end
    
    % Second half: 100-101 vs 110-111
    if prob_second_half > 0
        prob_10x = sum(probabilities(5:6));
        prob_11x = sum(probabilities(7:8));
        if prob_10x + prob_11x > 0
            theta_2 = 2 * asin(sqrt(prob_11x / (prob_10x + prob_11x)));
            state = apply_controlled_ry_gate(state, 1, 2, theta_2);  % control: qubit 0, target: qubit 1
            gate_count = gate_count + 1;
            circuit.gates{gate_count} = sprintf('CRY(%.4f) control=0, target=1', theta_2);
        end
    end
    
    % Level 3: Split individual states within each quarter
    quarters = {[1,2], [3,4], [5,6], [7,8]};
    control_patterns = {[0,0], [0,1], [1,0], [1,1]};
    
    for q = 1:4
        indices = quarters{q};
        prob_quarter = sum(probabilities(indices));
        
        if prob_quarter > 0 && probabilities(indices(1)) + probabilities(indices(2)) > 0
            theta = 2 * asin(sqrt(probabilities(indices(2)) / (probabilities(indices(1)) + probabilities(indices(2)))));
            
            % Apply doubly controlled rotation
            state = apply_ccry_gate(state, control_patterns{q}, theta);
            gate_count = gate_count + 1;
            circuit.gates{gate_count} = sprintf('CCRY(%.4f) controls=[%d,%d], target=2', ...
                theta, control_patterns{q}(1), control_patterns{q}(2));
        end
    end
    
    final_state = state;
    
    fprintf('Circuit has %d gates\n', gate_count);
end

function new_state = apply_ry_gate(state, qubit, angle)
    % Apply RY rotation to specified qubit
    % qubit: 1, 2, or 3 (1-indexed)
    
    % RY gate matrix
    RY = [cos(angle/2), -sin(angle/2); sin(angle/2), cos(angle/2)];
    
    % Create the full 8x8 gate matrix
    if qubit == 1
        % Qubit 0 (most significant)
        I = eye(2);
        gate_matrix = kron(kron(RY, I), I);
    elseif qubit == 2
        % Qubit 1 (middle)
        I = eye(2);
        gate_matrix = kron(kron(I, RY), I);
    else
        % Qubit 2 (least significant)
        I = eye(2);
        gate_matrix = kron(kron(I, I), RY);
    end
    
    new_state = gate_matrix * state;
end

function new_state = apply_controlled_ry_gate(state, control_qubit, target_qubit, angle)
    % Apply controlled RY gate
    % control_qubit: 1, 2, or 3
    % target_qubit: 1, 2, or 3
    
    % Start with identity
    gate_matrix = eye(8);
    
    % For each computational basis state, check if control is |1⟩
    for i = 1:8
        % Convert to binary representation (0-indexed)
        binary = dec2bin(i-1, 3) - '0';
        
        % Check if control qubit is 1
        if binary(control_qubit) == 1
            % Apply RY rotation to this state
            RY = [cos(angle/2), -sin(angle/2); sin(angle/2), cos(angle/2)];
            
            % Find the corresponding state with target qubit flipped
            binary_flipped = binary;
            binary_flipped(target_qubit) = 1 - binary_flipped(target_qubit);
            j = bin2dec(num2str(binary_flipped)) + 1;
            
            % Apply the rotation
            if binary(target_qubit) == 0
                gate_matrix(i, i) = cos(angle/2);
                gate_matrix(j, i) = sin(angle/2);
            else
                gate_matrix(i, i) = cos(angle/2);
                gate_matrix(j, i) = -sin(angle/2);
            end
        end
    end
    
    new_state = gate_matrix * state;
end

function new_state = apply_ccry_gate(state, control_pattern, angle)
    % Apply doubly controlled RY gate
    % control_pattern: [control1_value, control2_value] (0 or 1)
    % target is always qubit 2 (3rd qubit)
    
    gate_matrix = eye(8);
    
    for i = 1:8
        % Convert to binary representation
        binary = dec2bin(i-1, 3) - '0';
        
        % Check if both control qubits match the pattern
        if binary(1) == control_pattern(1) && binary(2) == control_pattern(2)
            % Apply RY rotation to target qubit (qubit 3)
            binary_flipped = binary;
            binary_flipped(3) = 1 - binary_flipped(3);
            j = bin2dec(num2str(binary_flipped)) + 1;
            
            if binary(3) == 0
                gate_matrix(i, i) = cos(angle/2);
                gate_matrix(j, i) = sin(angle/2);
            else
                gate_matrix(i, i) = cos(angle/2);
                gate_matrix(j, i) = -sin(angle/2);
            end
        end
    end
    
    new_state = gate_matrix * state;
end

function frequencies = simulate_measurements(state, num_shots)
    % Simulate quantum measurements
    % state: 8x1 complex vector
    % num_shots: number of measurement shots
    
    % Calculate probabilities
    probabilities = abs(state).^2;
    
    % Simulate measurements
    frequencies = zeros(1, 8);
    
    for shot = 1:num_shots
        % Random sample based on probabilities
        r = rand();
        cumulative = 0;
        
        for i = 1:8
            cumulative = cumulative + probabilities(i);
            if r <= cumulative
                frequencies(i) = frequencies(i) + 1;
                break;
            end
        end
    end
    
    % Normalize to frequencies
    frequencies = frequencies / num_shots;
end

function visualize_results(target, method1, method2)
    % Create visualization of results
    
    figure('Position', [100, 100, 800, 600]);
    
    % Create bar chart
    x = 1:8;
    width = 0.25;
    
    bar(x - width, target, width, 'FaceColor', [0.2, 0.6, 0.8], 'DisplayName', 'Target');
    hold on;
    bar(x, method1, width, 'FaceColor', [0.8, 0.4, 0.2], 'DisplayName', 'Method 1');
    bar(x + width, method2, width, 'FaceColor', [0.2, 0.8, 0.2], 'DisplayName', 'Method 2');
    
    % Customize plot
    xlabel('Quantum State');
    ylabel('Frequency');
    title('Quantum State Preparation Results');
    legend('Location', 'best');
    grid on;
    
    % Set x-axis labels
    labels = {'|000⟩', '|001⟩', '|010⟩', '|011⟩', '|100⟩', '|101⟩', '|110⟩', '|111⟩'};
    set(gca, 'XTick', 1:8, 'XTickLabel', labels);
    
    % Add error bars or annotations if needed
    ylim([0, max([target, method1, method2]) * 1.1]);
end

% Run the main function
quantum_state_preparation();