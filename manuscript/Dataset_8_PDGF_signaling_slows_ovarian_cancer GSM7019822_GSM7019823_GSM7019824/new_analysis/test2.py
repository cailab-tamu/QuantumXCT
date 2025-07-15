import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RYGate
from qiskit_aer import AerSimulator
from qiskit import transpile
import matplotlib.pyplot as plt

def create_state_preparation_circuit(frequencies):
    """
    Create a quantum circuit that prepares a state with given measurement frequencies.
    Uses a systematic approach with controlled rotations.
    """
    # Normalize frequencies to probabilities
    frequencies = np.array(frequencies)
    probabilities = frequencies / np.sum(frequencies)
    
    # Convert to amplitudes (assuming real amplitudes)
    amplitudes = np.sqrt(probabilities)
    
    print("Input frequencies:", frequencies)
    print("Normalized probabilities:", probabilities)
    print("Target amplitudes:", amplitudes)
    
    # Create quantum circuit with 3 qubits
    qc = QuantumCircuit(3, 3)
    
    # Method: Uniform Controlled Rotations
    # We'll use a systematic approach to prepare the state
    
    # First, let's group the amplitudes by qubit patterns
    # State |abc⟩ corresponds to amplitudes[4*a + 2*b + c]
    
    # Calculate rotation angles for each level
    # Level 1: Split probability between first 4 states (000-011) and last 4 states (100-111)
    prob_first_half = np.sum(probabilities[:4])
    prob_second_half = np.sum(probabilities[4:])
    
    if prob_first_half + prob_second_half > 0:
        theta_0 = 2 * np.arcsin(np.sqrt(prob_second_half))
        qc.ry(theta_0, 0)
    
    # Level 2: For each half, split between first 2 and last 2 states
    # First half: states 000-001 vs 010-011
    if prob_first_half > 0:
        prob_00x = np.sum(probabilities[0:2])
        prob_01x = np.sum(probabilities[2:4])
        if prob_00x + prob_01x > 0:
            theta_1 = 2 * np.arcsin(np.sqrt(prob_01x / (prob_00x + prob_01x)))
            qc.cry(theta_1, 0, 1)
    
    # Second half: states 100-101 vs 110-111
    if prob_second_half > 0:
        prob_10x = np.sum(probabilities[4:6])
        prob_11x = np.sum(probabilities[6:8])
        if prob_10x + prob_11x > 0:
            theta_2 = 2 * np.arcsin(np.sqrt(prob_11x / (prob_10x + prob_11x)))
            qc.cry(theta_2, 0, 1)
    
    # Level 3: For each quarter, split between individual states
    # We'll implement controlled-controlled-RY using basic gates
    
    # Quarter 1: 000 vs 001 (control on q0=0, q1=0)
    if np.sum(probabilities[0:2]) > 0:
        theta_3 = 2 * np.arcsin(np.sqrt(probabilities[1] / np.sum(probabilities[0:2])))
        qc.x(0)  # flip q0 to make it |1⟩
        qc.x(1)  # flip q1 to make it |1⟩
        qc.ccx(0, 1, 2)  # Toffoli gate as control
        qc.ry(theta_3, 2)  # Apply rotation
        qc.ccx(0, 1, 2)  # Undo Toffoli
        qc.x(1)  # restore q1
        qc.x(0)  # restore q0
    
    # Quarter 2: 010 vs 011 (control on q0=0, q1=1)
    if np.sum(probabilities[2:4]) > 0:
        theta_4 = 2 * np.arcsin(np.sqrt(probabilities[3] / np.sum(probabilities[2:4])))
        qc.x(0)  # flip q0 to make it |1⟩
        qc.ccx(0, 1, 2)  # Toffoli gate as control
        qc.ry(theta_4, 2)  # Apply rotation
        qc.ccx(0, 1, 2)  # Undo Toffoli
        qc.x(0)  # restore q0
    
    # Quarter 3: 100 vs 101 (control on q0=1, q1=0)
    if np.sum(probabilities[4:6]) > 0:
        theta_5 = 2 * np.arcsin(np.sqrt(probabilities[5] / np.sum(probabilities[4:6])))
        qc.x(1)  # flip q1 to make it |1⟩
        qc.ccx(0, 1, 2)  # Toffoli gate as control
        qc.ry(theta_5, 2)  # Apply rotation
        qc.ccx(0, 1, 2)  # Undo Toffoli
        qc.x(1)  # restore q1
    
    # Quarter 4: 110 vs 111 (control on q0=1, q1=1)
    if np.sum(probabilities[6:8]) > 0:
        theta_6 = 2 * np.arcsin(np.sqrt(probabilities[7] / np.sum(probabilities[6:8])))
        qc.ccx(0, 1, 2)  # Toffoli gate as control
        qc.ry(theta_6, 2)  # Apply rotation
        qc.ccx(0, 1, 2)  # Undo Toffoli
    
    return qc, amplitudes

def simulate_circuit(qc, shots=10000):
    """Simulate the quantum circuit and return measurement frequencies."""
    # Add measurements
    qc_with_measurement = qc.copy()
    qc_with_measurement.measure_all()
    
    # Simulate
    simulator = AerSimulator()
    compiled_circuit = transpile(qc_with_measurement, simulator)
    result = simulator.run(compiled_circuit, shots=shots).result()
    counts = result.get_counts()
    
    # Convert to frequency array
    frequencies = np.zeros(8)
    for state, count in counts.items():
        # Remove spaces and convert binary string to decimal index
        clean_state = state.replace(' ', '')
        # Reverse the string because Qiskit uses little-endian bit ordering
        # (rightmost bit is qubit 0)
        reversed_state = clean_state[::-1]
        index = int(reversed_state, 2)
        frequencies[index] = count / shots
    
    return frequencies

def alternative_method(frequencies):
    """
    Alternative method using a more direct approach with RY gates.
    This method systematically builds the superposition.
    """
    probabilities = np.array(frequencies) / np.sum(frequencies)
    
    qc = QuantumCircuit(3, 3)
    
    # More systematic approach using the uniform controlled rotations
    # This is a simplified version - for exact matching, we'd need optimization
    
    # Calculate cumulative probabilities for systematic preparation
    cumprobs = np.cumsum(probabilities)
    
    # Apply rotations to approximate the target distribution
    # This is a heuristic approach
    
    # First qubit rotation
    p_first_half = np.sum(probabilities[:4])
    if p_first_half < 1.0:
        angle = 2 * np.arccos(np.sqrt(p_first_half))
        qc.ry(angle, 0)
    
    # Second qubit rotations (controlled)
    p_00 = np.sum(probabilities[0:2])
    p_01 = np.sum(probabilities[2:4])
    if p_00 + p_01 > 0:
        angle = 2 * np.arccos(np.sqrt(p_00 / (p_00 + p_01)))
        qc.cry(angle, 0, 1)
    
    p_10 = np.sum(probabilities[4:6])
    p_11 = np.sum(probabilities[6:8])
    if p_10 + p_11 > 0:
        angle = 2 * np.arccos(np.sqrt(p_10 / (p_10 + p_11)))
        qc.x(0)
        qc.cry(angle, 0, 1)
        qc.x(0)
    
    # Third qubit rotations (doubly controlled using basic gates)
    # Helper function to implement controlled-controlled-RY
    def ccry(qc, angle, control1, control2, target):
        """Implement controlled-controlled-RY using basic gates"""
        qc.cry(angle/2, control2, target)
        qc.cx(control1, control2)
        qc.cry(-angle/2, control2, target)
        qc.cx(control1, control2)
        qc.cry(angle/2, control1, target)
    
    if probabilities[0] + probabilities[1] > 0:
        angle = 2 * np.arccos(np.sqrt(probabilities[0] / (probabilities[0] + probabilities[1])))
        qc.x(0)
        qc.x(1)
        ccry(qc, angle, 0, 1, 2)
        qc.x(1)
        qc.x(0)
    
    if probabilities[2] + probabilities[3] > 0:
        angle = 2 * np.arccos(np.sqrt(probabilities[2] / (probabilities[2] + probabilities[3])))
        qc.x(0)
        ccry(qc, angle, 0, 1, 2)
        qc.x(0)
    
    if probabilities[4] + probabilities[5] > 0:
        angle = 2 * np.arccos(np.sqrt(probabilities[4] / (probabilities[4] + probabilities[5])))
        qc.x(1)
        ccry(qc, angle, 0, 1, 2)
        qc.x(1)
    
    if probabilities[6] + probabilities[7] > 0:
        angle = 2 * np.arccos(np.sqrt(probabilities[6] / (probabilities[6] + probabilities[7])))
        ccry(qc, angle, 0, 1, 2)
    
    return qc


def extend_circuit_for_new_targets(original_circuit, original_freqs, new_target_freqs):
    """
    Extend the quantum circuit to transform from original frequency distribution 
    to new target frequency distribution.
    """
    # Start with the original circuit
    extended_qc = original_circuit.copy()
    
    # Convert frequencies to amplitudes
    original_probs = np.array(original_freqs) / np.sum(original_freqs)
    new_target_probs = np.array(new_target_freqs) / np.sum(new_target_freqs)
    
    original_amplitudes = np.sqrt(original_probs)
    new_target_amplitudes = np.sqrt(new_target_probs)
    
    print(f"Original amplitudes: {original_amplitudes}")
    print(f"New target amplitudes: {new_target_amplitudes}")
    
    # Add a separator comment
    extended_qc.barrier()
    
    # Method 1: Apply additional rotations to transform the distribution
    # This is a heuristic approach - we'll adjust probabilities systematically
    
    # Calculate the transformation needed for each level
    # Level 1: Adjust split between first half (000-011) and second half (100-111)
    original_first_half = np.sum(original_probs[:4])
    original_second_half = np.sum(original_probs[4:])
    
    new_first_half = np.sum(new_target_probs[:4])
    new_second_half = np.sum(new_target_probs[4:])
    
    # Calculate additional rotation needed
    if original_first_half > 0 and original_second_half > 0:
        original_ratio = original_second_half / (original_first_half + original_second_half)
        new_ratio = new_second_half / (new_first_half + new_second_half)
        
        # Apply correction rotation
        if abs(new_ratio - original_ratio) > 0.01:  # Only if significant difference
            correction_angle = 2 * (np.arcsin(np.sqrt(new_ratio)) - np.arcsin(np.sqrt(original_ratio)))
            extended_qc.ry(correction_angle, 0)
    
    # Level 2: Adjust splits within each half
    # First half adjustment
    if new_first_half > 0:
        original_00x = np.sum(original_probs[0:2])
        original_01x = np.sum(original_probs[2:4])
        new_00x = np.sum(new_target_probs[0:2])
        new_01x = np.sum(new_target_probs[2:4])
        
        if original_00x + original_01x > 0 and new_00x + new_01x > 0:
            original_ratio = original_01x / (original_00x + original_01x)
            new_ratio = new_01x / (new_00x + new_01x)
            
            if abs(new_ratio - original_ratio) > 0.01:
                correction_angle = 2 * (np.arcsin(np.sqrt(new_ratio)) - np.arcsin(np.sqrt(original_ratio)))
                extended_qc.cry(correction_angle, 0, 1)
    
    # Second half adjustment
    if new_second_half > 0:
        original_10x = np.sum(original_probs[4:6])
        original_11x = np.sum(original_probs[6:8])
        new_10x = np.sum(new_target_probs[4:6])
        new_11x = np.sum(new_target_probs[6:8])
        
        if original_10x + original_11x > 0 and new_10x + new_11x > 0:
            original_ratio = original_11x / (original_10x + original_11x)
            new_ratio = new_11x / (new_10x + new_11x)
            
            if abs(new_ratio - original_ratio) > 0.01:
                correction_angle = 2 * (np.arcsin(np.sqrt(new_ratio)) - np.arcsin(np.sqrt(original_ratio)))
                extended_qc.cry(correction_angle, 0, 1)
    
    # Level 3: Fine-tune individual state probabilities
    # This is more complex and requires careful handling of the controlled rotations
    
    # Apply fine-tuning rotations for each pair
    pairs = [(0,1), (2,3), (4,5), (6,7)]
    control_patterns = [(0,0), (0,1), (1,0), (1,1)]
    
    for i, (pair, controls) in enumerate(zip(pairs, control_patterns)):
        original_pair_sum = original_probs[pair[0]] + original_probs[pair[1]]
        new_pair_sum = new_target_probs[pair[0]] + new_target_probs[pair[1]]
        
        if original_pair_sum > 0 and new_pair_sum > 0:
            original_ratio = original_probs[pair[1]] / original_pair_sum
            new_ratio = new_target_probs[pair[1]] / new_pair_sum
            
            if abs(new_ratio - original_ratio) > 0.01:
                correction_angle = 2 * (np.arcsin(np.sqrt(new_ratio)) - np.arcsin(np.sqrt(original_ratio)))
                
                # Apply controlled rotation based on the control pattern
                if controls == (0,0):
                    extended_qc.x(0)
                    extended_qc.x(1)
                    apply_ccry_to_circuit(extended_qc, correction_angle, 0, 1, 2)
                    extended_qc.x(1)
                    extended_qc.x(0)
                elif controls == (0,1):
                    extended_qc.x(0)
                    apply_ccry_to_circuit(extended_qc, correction_angle, 0, 1, 2)
                    extended_qc.x(0)
                elif controls == (1,0):
                    extended_qc.x(1)
                    apply_ccry_to_circuit(extended_qc, correction_angle, 0, 1, 2)
                    extended_qc.x(1)
                else:  # (1,1)
                    apply_ccry_to_circuit(extended_qc, correction_angle, 0, 1, 2)
    
    return extended_qc

def apply_ccry_to_circuit(qc, angle, control1, control2, target):
    """Apply controlled-controlled-RY gate to circuit using basic gates"""
    # Decompose CCRY using available gates
    qc.cry(angle/2, control2, target)
    qc.cx(control1, control2)
    qc.cry(-angle/2, control2, target)
    qc.cx(control1, control2)
    qc.cry(angle/2, control1, target)


# Your input frequencies
original_frequencies = [0.2346, 0.1829, 0.0911, 0.0617, 0.1646, 0.1259, 0.0807, 0.0585]
new_target_frequencies = [0.2402, 0.0619, 0.1443, 0.0355, 0.2304, 0.0476, 0.1971, 0.0431]

# Create the quantum circuit for original frequencies
print("Creating quantum state preparation circuit for original frequencies...")
qc_original, original_amplitudes = create_state_preparation_circuit(original_frequencies)

print(f"\nOriginal Quantum Circuit:")
print(qc_original)

# Now extend the circuit to transform to new target frequencies
print("\nExtending circuit to transform to new target frequencies...")
qc_extended = extend_circuit_for_new_targets(qc_original, original_frequencies, new_target_frequencies)

print(f"\nExtended Quantum Circuit:")
print(qc_extended)

# Simulate both circuits
print("\nSimulating original circuit...")
measured_frequencies_original = simulate_circuit(qc_original, shots=10000)

print("Simulating extended circuit...")
measured_frequencies_extended = simulate_circuit(qc_extended, shots=10000)

# Compare all results
print("\nComparison of all frequencies:")
print("State | Original | New Target | Extended | Orig Error | New Error")
print("------|----------|------------|----------|------------|----------")
for i in range(8):
    binary = format(i, '03b')
    orig_error = abs(original_frequencies[i] - measured_frequencies_original[i])
    new_error = abs(new_target_frequencies[i] - measured_frequencies_extended[i])
    print(f"|{binary}⟩  | {original_frequencies[i]:.4f}   | {new_target_frequencies[i]:.4f}     | {measured_frequencies_extended[i]:.4f}   | {orig_error:.4f}     | {new_error:.4f}")

total_error_original = np.sum(np.abs(np.array(original_frequencies) - measured_frequencies_original))
total_error_extended = np.sum(np.abs(np.array(new_target_frequencies) - measured_frequencies_extended))

print(f"\nTotal absolute errors:")
print(f"Original circuit: {total_error_original:.4f}")
print(f"Extended circuit: {total_error_extended:.4f}")

# Show the transformation
print(f"\nTransformation Analysis:")
print("State | Change    | Relative Change")
print("------|-----------|----------------")
for i in range(8):
    binary = format(i, '03b')
    change = new_target_frequencies[i] - original_frequencies[i]
    rel_change = change / original_frequencies[i] * 100 if original_frequencies[i] > 0 else float('inf')
    print(f"|{binary}⟩  | {change:+.4f}   | {rel_change:+6.1f}%")
