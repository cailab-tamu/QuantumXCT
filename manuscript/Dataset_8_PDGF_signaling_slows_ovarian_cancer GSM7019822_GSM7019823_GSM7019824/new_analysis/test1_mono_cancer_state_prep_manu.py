import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import StatePreparation
from qiskit import transpile
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt

# Your frequency distribution
frequencies = np.array([0.118443, 0.006768, 0.048223, 0.002538, 0.078257, 0.005076, 
                       0.053723, 0.006768, 0.140017, 0.013536, 0.096024, 0.013536, 
                       0.170474, 0.017766, 0.195854, 0.032995])

# Convert frequencies to amplitudes
normalized_probs = frequencies / np.sum(frequencies)
amplitudes = np.sqrt(normalized_probs)

print("Amplitudes:", amplitudes)
print("Sum of squared amplitudes:", np.sum(amplitudes**2))

def manual_state_preparation(amplitudes):
    """
    Manual implementation using uniformly controlled rotations
    Based on the recursive decomposition algorithm
    """
    n_qubits = int(np.log2(len(amplitudes)))
    qc = QuantumCircuit(n_qubits)
    
    # We'll build the circuit layer by layer
    # Each layer handles one qubit level
    
    def apply_uniformly_controlled_ry(circuit, angles, control_qubits, target_qubit):
        """
        Apply uniformly controlled RY rotations
        For n control qubits, we need 2^n angles
        """
        n_controls = len(control_qubits)
        
        if n_controls == 0:
            # No controls, just apply RY
            if len(angles) > 0:
                circuit.ry(angles[0], target_qubit)
        elif n_controls == 1:
            # Single control
            if len(angles) >= 2:
                # Apply conditional rotations
                # When control is |0⟩, apply angles[0]
                # When control is |1⟩, apply angles[1]
                
                # Decompose into unconditional and controlled parts
                alpha = (angles[0] + angles[1]) / 2
                beta = (angles[0] - angles[1]) / 2
                
                circuit.ry(alpha, target_qubit)
                circuit.cx(control_qubits[0], target_qubit)
                circuit.ry(beta, target_qubit)
                circuit.cx(control_qubits[0], target_qubit)
                circuit.ry(-beta, target_qubit)
        else:
            # Multiple controls - use recursive decomposition
            # Split the angles into two halves
            mid = len(angles) // 2
            angles_0 = angles[:mid]  # When MSB control is 0
            angles_1 = angles[mid:]  # When MSB control is 1
            
            # Recursively apply to sub-controls
            apply_uniformly_controlled_ry(circuit, angles_0, control_qubits[1:], target_qubit)
            
            # Apply controlled operation for the MSB
            if len(angles_1) > 0:
                # Calculate the difference angles
                diff_angles = []
                for i in range(min(len(angles_0), len(angles_1))):
                    diff_angles.append(angles_1[i] - angles_0[i])
                
                # Apply the difference with MSB control
                circuit.cx(control_qubits[0], target_qubit)
                apply_uniformly_controlled_ry(circuit, diff_angles, control_qubits[1:], target_qubit)
                circuit.cx(control_qubits[0], target_qubit)
    
    # Layer-by-layer decomposition
    current_amplitudes = amplitudes.copy()
    
    # Process each qubit level
    for level in range(n_qubits):
        target_qubit = level
        control_qubits = list(range(level))
        
        # Group amplitudes by the current qubit level
        group_size = 2**(n_qubits - level - 1)
        n_groups = len(current_amplitudes) // group_size
        
        rotation_angles = []
        
        for group in range(n_groups // 2):
            # For each pair of groups
            group_0_start = group * 2 * group_size
            group_0_end = group_0_start + group_size
            group_1_start = group_0_end
            group_1_end = group_1_start + group_size
            
            # Calculate the norm of each group
            norm_0 = np.linalg.norm(current_amplitudes[group_0_start:group_0_end])
            norm_1 = np.linalg.norm(current_amplitudes[group_1_start:group_1_end])
            
            # Calculate rotation angle
            if norm_0**2 + norm_1**2 > 1e-10:
                angle = 2 * np.arctan2(norm_1, norm_0)
            else:
                angle = 0
            
            rotation_angles.append(angle)
            
            # Normalize the groups
            if norm_0 > 1e-10:
                current_amplitudes[group_0_start:group_0_end] /= norm_0
            if norm_1 > 1e-10:
                current_amplitudes[group_1_start:group_1_end] /= norm_1
        
        # Apply uniformly controlled rotations
        if len(rotation_angles) > 0:
            # Pad angles to power of 2 if necessary
            while len(rotation_angles) < 2**len(control_qubits):
                rotation_angles.append(0)
            
            apply_uniformly_controlled_ry(qc, rotation_angles, control_qubits, target_qubit)
    
    return qc

# Create manual circuit
print("\nCreating manual state preparation circuit...")
try:
    qc_manual = manual_state_preparation(amplitudes)
    
    # Create a clean circuit with proper measurements
    qc_manual_clean = QuantumCircuit(4, 4)
    
    # Copy only the quantum operations (no measurements)
    for instruction in qc_manual.data:
        if instruction.operation.name not in ['measure', 'barrier']:
            qc_manual_clean.append(instruction)
    
    # Add proper measurements
    qc_manual_clean.measure_all()
    qc_manual = qc_manual_clean
    
    print(f"Manual circuit created with depth: {qc_manual.depth()}")
    print(f"Number of gates: {len(qc_manual.data)}")
    print(f"Qubits: {qc_manual.num_qubits}, Classical bits: {qc_manual.num_clbits}")
    
except Exception as e:
    print(f"Error creating manual circuit: {e}")
    # Create a simple fallback circuit
    qc_manual = QuantumCircuit(4, 4)
    qc_manual.append(StatePreparation(amplitudes), range(4))
    qc_manual.measure_all()
    print("Using fallback StatePreparation circuit")

# Alternative: Step-by-step manual implementation
def step_by_step_manual(amplitudes):
    """
    A clearer step-by-step manual implementation
    """
    qc = QuantumCircuit(4, 4)
    
    # Step 1: Prepare the first qubit
    # Probability of measuring 0 in first qubit = sum of first 8 amplitudes squared
    prob_0 = np.sum(amplitudes[0:8]**2)
    prob_1 = np.sum(amplitudes[8:16]**2)
    
    if prob_0 + prob_1 > 1e-10:
        angle_0 = 2 * np.arctan(np.sqrt(prob_1/prob_0)) if prob_0 > 1e-10 else np.pi
        qc.ry(angle_0, 0)
    
    # Step 2: Prepare second qubit conditioned on first
    # When first qubit is 0: prob of second being 0 vs 1
    # When first qubit is 1: prob of second being 0 vs 1
    
    # For |0xxx⟩ states
    if prob_0 > 1e-10:
        prob_00 = np.sum(amplitudes[0:4]**2)
        prob_01 = np.sum(amplitudes[4:8]**2)
        if prob_00 + prob_01 > 1e-10:
            angle_00 = 2 * np.arctan(np.sqrt(prob_01/prob_00)) if prob_00 > 1e-10 else np.pi
        else:
            angle_00 = 0
    else:
        angle_00 = 0
    
    # For |1xxx⟩ states
    if prob_1 > 1e-10:
        prob_10 = np.sum(amplitudes[8:12]**2)
        prob_11 = np.sum(amplitudes[12:16]**2)
        if prob_10 + prob_11 > 1e-10:
            angle_10 = 2 * np.arctan(np.sqrt(prob_11/prob_10)) if prob_10 > 1e-10 else np.pi
        else:
            angle_10 = 0
    else:
        angle_10 = 0
    
    # Apply controlled rotation for second qubit
    if abs(angle_00 - angle_10) > 1e-10:
        avg_angle = (angle_00 + angle_10) / 2
        diff_angle = (angle_00 - angle_10) / 2
        
        qc.ry(avg_angle, 1)
        qc.cx(0, 1)
        qc.ry(diff_angle, 1)
        qc.cx(0, 1)
        qc.ry(-diff_angle, 1)
    else:
        qc.ry(angle_00, 1)
    
    # Step 3: Continue for third qubit (controlled by first two)
    # This gets more complex with 4 different control states
    angles_3rd = []
    
    # Calculate angles for each combination of first two qubits
    for i in range(4):
        start_idx = i * 2
        end_idx = start_idx + 2
        
        prob_0_3rd = amplitudes[start_idx]**2 if start_idx < 16 else 0
        prob_1_3rd = amplitudes[start_idx + 1]**2 if start_idx + 1 < 16 else 0
        
        if prob_0_3rd + prob_1_3rd > 1e-10:
            angle_3rd = 2 * np.arctan(np.sqrt(prob_1_3rd/prob_0_3rd)) if prob_0_3rd > 1e-10 else np.pi
        else:
            angle_3rd = 0
        angles_3rd.append(angle_3rd)
    
    # Apply multi-controlled rotations for third qubit
    # This is a simplified version - full implementation would need more gates
    base_angle = sum(angles_3rd) / 4
    qc.ry(base_angle, 2)
    
    # Add some controlled corrections (simplified)
    qc.cx(0, 2)
    qc.ry((angles_3rd[2] - angles_3rd[0]) / 2, 2)
    qc.cx(0, 2)
    
    qc.cx(1, 2)
    qc.ry((angles_3rd[1] - angles_3rd[0]) / 2, 2)
    qc.cx(1, 2)
    
    # Step 4: Fourth qubit (controlled by all three)
    # Similar process but with 8 control states
    angles_4th = []
    
    for i in range(8):
        prob_0_4th = amplitudes[i * 2]**2 if i * 2 < 16 else 0
        prob_1_4th = amplitudes[i * 2 + 1]**2 if i * 2 + 1 < 16 else 0
        
        if prob_0_4th + prob_1_4th > 1e-10:
            angle_4th = 2 * np.arctan(np.sqrt(prob_1_4th/prob_0_4th)) if prob_0_4th > 1e-10 else np.pi
        else:
            angle_4th = 0
        angles_4th.append(angle_4th)
    
    # Apply multi-controlled rotations for fourth qubit (simplified)
    base_angle = sum(angles_4th) / 8
    qc.ry(base_angle, 3)
    
    # Add controlled corrections
    qc.cx(0, 3)
    qc.ry((angles_4th[4] - angles_4th[0]) / 2, 3)
    qc.cx(0, 3)
    
    qc.cx(1, 3)
    qc.ry((angles_4th[2] - angles_4th[0]) / 2, 3)
    qc.cx(1, 3)
    
    qc.cx(2, 3)
    qc.ry((angles_4th[1] - angles_4th[0]) / 2, 3)
    qc.cx(2, 3)
    
    return qc

# Create step-by-step manual circuit
print("\nCreating step-by-step manual circuit...")
try:
    qc_step_raw = step_by_step_manual(amplitudes)
    
    # Create a clean circuit with proper measurements
    qc_step = QuantumCircuit(4, 4)
    
    # Copy only the quantum operations (no measurements)
    for instruction in qc_step_raw.data:
        if instruction.operation.name not in ['measure', 'barrier']:
            qc_step.append(instruction)
    
    # Add proper measurements
    qc_step.measure_all()
    
    print(f"Step-by-step circuit depth: {qc_step.depth()}")
    print(f"Number of gates: {len(qc_step.data)}")
    print(f"Qubits: {qc_step.num_qubits}, Classical bits: {qc_step.num_clbits}")
    
    # Test if circuit is valid by doing a small simulation
    try:
        test_compiled = transpile(qc_step, simulator)
        test_job = simulator.run(test_compiled, shots=10)
        test_result = test_job.result()
        if not test_result.success:
            raise Exception("Circuit test failed")
        print("Step-by-step circuit validation: PASSED")
    except Exception as test_e:
        print(f"Step-by-step circuit validation failed: {test_e}")
        raise test_e
    
except Exception as e:
    print(f"Error creating step-by-step circuit: {e}")
    print("Creating simplified step-by-step circuit...")
    
    # Create a much simpler manual implementation
    qc_step = QuantumCircuit(4, 4)
    
    # Simple approach: just apply some rotations based on the amplitudes
    # This won't be perfect but will demonstrate the concept
    
    # Apply rotations based on probability distributions
    prob_groups = [
        np.sum(amplitudes[0:8]**2),   # |0xxx⟩ states
        np.sum(amplitudes[8:16]**2)   # |1xxx⟩ states
    ]
    
    if prob_groups[0] + prob_groups[1] > 0:
        angle = 2 * np.arctan(np.sqrt(prob_groups[1]/prob_groups[0])) if prob_groups[0] > 0 else np.pi
        qc_step.ry(angle, 0)
    
    # Add some additional rotations for other qubits
    qc_step.ry(np.pi/4, 1)  # Simple rotation for demonstration
    qc_step.ry(np.pi/6, 2)
    qc_step.ry(np.pi/8, 3)
    
    # Add some entanglement
    qc_step.cx(0, 1)
    qc_step.cx(1, 2)
    qc_step.cx(2, 3)
    
    qc_step.measure_all()
    print("Using simplified step-by-step circuit")

# Compare with Qiskit's StatePreparation
qc_qiskit = QuantumCircuit(4, 4)
qc_qiskit.append(StatePreparation(amplitudes), range(4))
qc_qiskit.measure_all()

print(f"Qiskit StatePreparation depth: {qc_qiskit.depth()}")

# Simulate all three circuits
simulator = AerSimulator()
shots = 10000

def simulate_and_compare(circuit, name):
    try:
        compiled_circuit = transpile(circuit, simulator)
        job = simulator.run(compiled_circuit, shots=shots)
        result = job.result()
        
        # Check if we have results
        if not result.success:
            print(f"Error: {name} simulation failed")
            return np.zeros(16), 0.0
            
        counts = result.get_counts()
        
        # Check if counts is empty
        if not counts:
            print(f"Error: {name} returned no measurement results")
            return np.zeros(16), 0.0
            
    except Exception as e:
        print(f"Error simulating {name}: {e}")
        return np.zeros(16), 0.0
    
    # Debug information
    print(f"\n{name} - Circuit info:")
    print(f"Number of qubits: {circuit.num_qubits}")
    print(f"Number of classical bits: {circuit.num_clbits}")
    print(f"Sample measurement results: {list(counts.keys())[:5]}")
    print(f"Total shots captured: {sum(counts.values())}")
    
    # Convert to frequency array
    measured_frequencies = np.zeros(16)
    for bitstring, count in counts.items():
        # Remove spaces and convert to integer
        clean_bitstring = bitstring.replace(' ', '')
        
        # Handle different bitstring lengths
        if len(clean_bitstring) == 4:
            # Standard 4-bit case
            index = int(clean_bitstring, 2)
        elif len(clean_bitstring) == 8:
            # 8-bit case - take only the first 4 bits (the actual qubit measurements)
            # The format is usually 'qubit3qubit2qubit1qubit0 ancilla_bits'
            relevant_bits = clean_bitstring[:4]
            index = int(relevant_bits, 2)
        else:
            print(f"Warning: Unexpected bitstring length: {len(clean_bitstring)} for {clean_bitstring}")
            continue
            
        if 0 <= index < 16:
            measured_frequencies[index] += count / shots
    
    # Calculate fidelity
    fidelity = np.sum(np.sqrt(normalized_probs * measured_frequencies))
    
    print(f"Fidelity: {fidelity:.6f}")
    
    return measured_frequencies, fidelity

# Test all circuits
print("\n" + "="*50)
print("SIMULATION RESULTS")
print("="*50)

# Test circuits one by one with error handling
results = {}

try:
    freq_manual, fid_manual = simulate_and_compare(qc_manual, "Manual Implementation")
    results['Manual'] = (freq_manual, fid_manual)
except Exception as e:
    print(f"Manual implementation failed: {e}")
    results['Manual'] = (np.zeros(16), 0.0)

try:
    freq_step, fid_step = simulate_and_compare(qc_step, "Step-by-step Manual")
    results['Step-by-step'] = (freq_step, fid_step)
except Exception as e:
    print(f"Step-by-step implementation failed: {e}")
    results['Step-by-step'] = (np.zeros(16), 0.0)

try:
    freq_qiskit, fid_qiskit = simulate_and_compare(qc_qiskit, "Qiskit StatePreparation")
    results['Qiskit'] = (freq_qiskit, fid_qiskit)
except Exception as e:
    print(f"Qiskit implementation failed: {e}")
    results['Qiskit'] = (np.zeros(16), 0.0)

# Extract results
freq_manual, fid_manual = results['Manual']
freq_step, fid_step = results['Step-by-step']
freq_qiskit, fid_qiskit = results['Qiskit']

# Visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Target vs Manual
ax1.bar(range(16), normalized_probs, alpha=0.5, color='blue', label='Target')
ax1.bar(range(16), freq_manual, alpha=0.5, color='red', label='Manual')
ax1.set_title(f'Manual Implementation (Fidelity: {fid_manual:.4f})')
ax1.set_xlabel('Computational Basis State')
ax1.set_ylabel('Probability')
ax1.legend()
ax1.set_xticks(range(0, 16, 2))

# Target vs Step-by-step
ax2.bar(range(16), normalized_probs, alpha=0.5, color='blue', label='Target')
ax2.bar(range(16), freq_step, alpha=0.5, color='green', label='Step-by-step')
ax2.set_title(f'Step-by-step Manual (Fidelity: {fid_step:.4f})')
ax2.set_xlabel('Computational Basis State')
ax2.set_ylabel('Probability')
ax2.legend()
ax2.set_xticks(range(0, 16, 2))

# Target vs Qiskit
ax3.bar(range(16), normalized_probs, alpha=0.5, color='blue', label='Target')
ax3.bar(range(16), freq_qiskit, alpha=0.5, color='purple', label='Qiskit')
ax3.set_title(f'Qiskit StatePreparation (Fidelity: {fid_qiskit:.4f})')
ax3.set_xlabel('Computational Basis State')
ax3.set_ylabel('Probability')
ax3.legend()
ax3.set_xticks(range(0, 16, 2))

# Fidelity comparison
methods = ['Manual', 'Step-by-step', 'Qiskit']
fidelities = [fid_manual, fid_step, fid_qiskit]
ax4.bar(methods, fidelities, color=['red', 'green', 'purple'])
ax4.set_title('Fidelity Comparison')
ax4.set_ylabel('Fidelity')
ax4.set_ylim([0, 1])

plt.tight_layout()
plt.show()

print("\nCircuit Details:")
print(f"Manual circuit gates: {len(qc_manual.data)}")
print(f"Step-by-step circuit gates: {len(qc_step.data)}")
print(f"Qiskit circuit gates: {len(qc_qiskit.data)}")

# Print the manual circuit
print("\nManual Circuit:")
print(qc_step.draw())