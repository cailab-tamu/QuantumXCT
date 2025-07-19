import numpy as np
import os
import json
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import StatePreparation
from qiskit import transpile
from qiskit_aer import AerSimulator

# Test file saving at the beginning
script_dir = os.path.dirname(os.path.abspath(__file__))
test_file = os.path.join(script_dir, 'test_write_access.json')
try:
    with open(test_file, 'w') as f:
        json.dump({'test': 'write_access_test'}, f)
    os.remove(test_file)  # Clean up
    print("✓ Write access verified in directory:", script_dir)
except Exception as e:
    print("✗ Cannot write to directory:", script_dir)
    print("Error:", str(e))
    print("Please check directory permissions and try again.")
    exit(1)
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Target frequency vectors
pt_f_mo = [0.234577, 0.182927, 0.091105, 0.061693, 0.164634, 0.125897, 0.080703, 0.058465]
pt_c_mo = [0.118443, 0.006768, 0.048223, 0.002538, 0.078257, 0.005076, 0.053723, 0.006768, 0.140017, 0.013536, 0.096024, 0.013536, 0.170474, 0.017766, 0.195854, 0.032995]

# Target frequency vectors for optimization
pt_f_co = np.array([0.240181, 0.061934, 0.144260, 0.035498, 0.230363, 0.047583, 0.197130, 0.043051])
pt_c_co = np.array([0.115983, 0.019802, 0.053748, 0.019095, 0.077793, 0.022631, 0.056577, 0.025460, 
                    0.083451, 0.026167, 0.062235, 0.034653, 0.108911, 0.049505, 0.142150, 0.101839])

# Normalize target frequencies to ensure they sum to 1
pt_f_co = pt_f_co / np.sum(pt_f_co)
pt_c_co = pt_c_co / np.sum(pt_c_co)

frequencies = np.array(pt_f_mo)
normalized_probs = frequencies / np.sum(frequencies)
amplitudes1 = np.sqrt(normalized_probs)
qreg1 = QuantumRegister(3, 'q')
creg1 = ClassicalRegister(3, 'c')
qc1 = QuantumCircuit(qreg1, creg1)

state_prep = StatePreparation(amplitudes1)
qc1.append(state_prep, qreg1)
qc1.measure(qreg1, creg1)
print(f"\nQuantum Circuit (depth: {qc1.depth()}):")
print(qc1)


frequencies = np.array(pt_c_mo)
normalized_probs = frequencies / np.sum(frequencies)
amplitudes2 = np.sqrt(normalized_probs)
qreg2 = QuantumRegister(4, 'q')
creg2 = ClassicalRegister(4, 'c')
qc2 = QuantumCircuit(qreg2, creg2)

state_prep2 = StatePreparation(amplitudes2)
qc2.append(state_prep2, qreg2)
qc2.measure(qreg2, creg2)
print(f"\nQuantum Circuit (depth: {qc2.depth()}):")
print(qc2)

def create_entangled_circuit(qc1, qc2, thetas=None):
    """
    Create a circuit with CRY gates between two quantum circuits.
    
    Args:
        qc1: First quantum circuit (3 qubits)
        qc2: Second quantum circuit (4 qubits)
        thetas: List of 24 angles (in radians) for the CRY gates.
               First 12: from qc1 to qc2 (3*4=12)
               Next 12: from qc2 to qc1 (4*3=12, excluding self-loops)
               If None, uses pi/4 for all angles.
    
    Returns:
        QuantumCircuit: The combined circuit with CRY gates
    """
    # Merge the circuits
    merged_circuit = qc1 ^ qc2
    print(f"Merged circuit has {merged_circuit.num_qubits} qubits")  # Should be 7
    
    # Initialize thetas if not provided
    if thetas is None:
        thetas = [np.pi/4] * 24  # Default to pi/4 for all thetas
    elif len(thetas) != 24:
        raise ValueError("thetas must be a list of 24 values")
    
    # Add CRY gates from first 3 qubits to last 4 qubits
    theta_idx = 0
    for i in range(3):  # First 3 qubits (0,1,2)
        for j in range(3, 7):  # Last 4 qubits (3,4,5,6)
            merged_circuit.cry(thetas[theta_idx], i, j)
            theta_idx += 1
    
    # Add CRY gates from last 4 qubits to first 3 qubits
    for i in range(3, 7):  # Last 4 qubits (3,4,5,6)
        for j in range(3):  # First 3 qubits (0,1,2)
            if i != j:  # Avoid self-loops
                merged_circuit.cry(thetas[theta_idx], i, j)
                theta_idx += 1
    
    return merged_circuit, thetas

def print_theta_info(thetas):
    """Print information about the theta values used for CRY gates."""
    print("\nTheta values used for CRY gates:")
    print("From first 3 qubits to last 4 qubits (12 gates):")
    for i in range(12):
        print(f"  Theta {i+1:2d} (q{i//4}->q{i%4+3}): {thetas[i]:.4f}")
    
    print("\nFrom last 4 qubits to first 3 qubits (12 gates):")
    for i in range(12, 24):
        q_from = 3 + (i-12) // 3
        q_to = (i-12) % 3
        print(f"  Theta {i+1:2d} (q{q_from}->q{q_to}): {thetas[i]:.4f}")

def kl_divergence(p, q, epsilon=1e-10):
    """
    Calculate KL divergence between two probability distributions.
    
    Args:
        p: Target probability distribution
        q: Approximate probability distribution
        epsilon: Small value to avoid log(0)
        
    Returns:
        float: KL divergence D_KL(p || q)
    """
    # Ensure we don't have zeros by adding a small value
    p_safe = np.clip(p, epsilon, 1)
    q_safe = np.clip(q, epsilon, 1)
    
    # Normalize to ensure they sum to 1
    p_norm = p_safe / np.sum(p_safe)
    q_norm = q_safe / np.sum(q_safe)
    
    # Calculate KL divergence
    return np.sum(p_norm * np.log(p_norm / q_norm))

def calculate_cost(thetas, qc1, qc2, shots=10000):
    """
    Calculate the cost using KL divergence between measured and target frequencies.
    
    Args:
        thetas: List of 24 angles for CRY gates
        qc1: First quantum circuit (3 qubits)
        qc2: Second quantum circuit (4 qubits)
        shots: Number of shots for simulation
        
    Returns:
        float: Total cost (KL divergence between measured and target frequencies)
    """
    # Create circuit with current thetas
    circuit, _ = create_entangled_circuit(qc1, qc2, thetas)
    
    # Add measurements
    measured_circuit = circuit.copy()
    cr = ClassicalRegister(7, 'meas')
    measured_circuit.add_register(cr)
    measured_circuit.measure(range(7), range(7))
    
    # Simulate the circuit
    simulator = AerSimulator()
    compiled_circuit = transpile(measured_circuit, simulator)
    result = simulator.run(compiled_circuit, shots=shots).result()
    counts = result.get_counts()
    
    # Initialize frequency arrays
    measured_f3 = np.zeros(8)
    measured_c4 = np.zeros(16)
    total_shots = sum(counts.values())
    
    # Count frequencies
    for state, count in counts.items():
        state_rev = state[::-1]  # Reverse for Qiskit's little-endian
        first3 = int(state_rev[:3], 2)  # Convert binary string to integer
        last4 = int(state_rev[3:7], 2)  # Convert binary string to integer
        measured_f3[first3] += count / total_shots
        measured_c4[last4] += count / total_shots
    
    # Calculate KL divergence for both distributions
    kl_f3 = kl_divergence(pt_f_co, measured_f3)
    kl_c4 = kl_divergence(pt_c_co, measured_c4)
    
    # Total cost is the sum of both KL divergences
    total_cost = kl_f3 + kl_c4
    
    return total_cost

def optimize_thetas(initial_thetas, qc1, qc2, maxiter=50):
    """
    Optimize thetas to match target frequencies.
    
    Args:
        initial_thetas: Initial values for the 24 thetas
        qc1: First quantum circuit (3 qubits)
        qc2: Second quantum circuit (4 qubits)
        maxiter: Maximum number of iterations for optimization
        
    Returns:
        tuple: (optimized_thetas, result) where optimized_thetas is the list of optimized angles
               and result is the full optimization result object
    """
    # Bounds for thetas (0 to 2π)
    bounds = [(0, 2*np.pi) for _ in range(24)]
    
    # Optimize using COBYLA method
    result = minimize(
        lambda x: calculate_cost(x, qc1, qc2),
        initial_thetas,
        method='COBYLA',
        options={'maxiter': maxiter, 'disp': True}
    )
    
    return result.x, result

# Example usage with optimization
print("\nStarting optimization to match target frequencies...")
initial_thetas = [np.pi/4] * 24  # Initial guess
optimized_thetas, result = optimize_thetas(initial_thetas, qc1, qc2)

print("\nOptimization results:")
print(f"Success: {result.success}")
print(f"Final cost: {result.fun:.6f}")
print(f"Number of iterations: {result.nfev}")

# Create circuit with optimized thetas
print("\nCreating circuit with optimized thetas:")
optimized_circuit, _ = create_entangled_circuit(qc1, qc2, optimized_thetas)
print_theta_info(optimized_thetas)

# Create a copy of the circuit for measurement
measured_circuit = optimized_circuit.copy()

# Add classical registers for measurements
cr = ClassicalRegister(7, 'meas')
measured_circuit.add_register(cr)

# Measure all qubits
measured_circuit.measure(range(7), range(7))

# Print the circuit to verify measurements
print("\nCircuit with measurements:")
print(measured_circuit)

# Simulate the circuit
simulator = AerSimulator()
compiled_circuit = transpile(measured_circuit, simulator)
result = simulator.run(compiled_circuit, shots=10000).result()
counts = result.get_counts()

# Initialize frequency dictionaries
first3_freq = {format(i, '03b'): 0 for i in range(8)}
last4_freq = {format(i, '04b'): 0 for i in range(16)}

total_shots = sum(counts.values())

# Process the counts
for state, count in counts.items():
    # The state string is in Qiskit's little-endian format (q0 is rightmost bit)
    # So we need to reverse the string to get the correct bit order
    state_rev = state[::-1]  # Reverse the state string
    
    # Get first 3 qubits (q0, q1, q2) - these are now the first 3 bits from the left
    first3 = state_rev[:3]
    # Get last 4 qubits (q3, q4, q5, q6) - these are the next 4 bits
    last4 = state_rev[3:7]
    
    # Update frequencies
    first3_freq[first3] += count / total_shots
    last4_freq[last4] += count / total_shots

def print_frequencies(freq_dict, title, states_per_row=4):
    print(f"\n{title}")
    print("-" * 80)
    states = sorted(freq_dict.items())
    for i in range(0, len(states), states_per_row):
        batch = states[i:i+states_per_row]
        # Print state labels
        print("  ".join(f"|{s[0]}⟩".ljust(8) for s in batch))
        # Print frequencies
        print("  ".join(f"{s[1]:.4f}".ljust(8) for s in batch))
        print()  # Add empty line between rows

# Print frequencies in a clean format
print_frequencies(first3_freq, "FREQUENCIES FOR FIRST 3 QUBITS (8 STATES)", states_per_row=4)
print_frequencies(last4_freq, "FREQUENCIES FOR LAST 4 QUBITS (16 STATES)", states_per_row=4)

# Print some statistics
print("\nSTATISTICS:")
print("-" * 80)
print(f"Total probability (first 3 qubits): {sum(first3_freq.values()):.4f}")
print(f"Total probability (last 4 qubits):  {sum(last4_freq.values()):.4f}")

# Find most probable states
most_probable_first3 = max(first3_freq.items(), key=lambda x: x[1])
most_probable_last4 = max(last4_freq.items(), key=lambda x: x[1])
print(f"\nMost probable state for first 3 qubits: |{most_probable_first3[0]}⟩ (p = {most_probable_first3[1]:.4f})")
print(f"Most probable state for last 4 qubits:  |{most_probable_last4[0]}⟩ (p = {most_probable_last4[1]:.4f})")


# After the simulation result, add this code:

# Print optimized thetas in a clear format
print("\n" + "="*80)
print("OPTIMIZED THETA VALUES (in radians):")
print("="*80)
print("\nFirst 12 thetas (from first 3 qubits to last 4 qubits):")
for i in range(12):
    print(f"Theta {i+1:2d} (q{i//4}->q{i%4+3}): {optimized_thetas[i]:.6f}")

print("\nNext 12 thetas (from last 4 qubits to first 3 qubits):")
for i in range(12, 24):
    q_from = 3 + (i-12) // 3
    q_to = (i-12) % 3
    print(f"Theta {i+1:2d} (q{q_from}->q{q_to}): {optimized_thetas[i]:.6f}")

# Save thetas to a file
import json
import os

theta_dict = {
    'thetas': optimized_thetas.tolist() if hasattr(optimized_thetas, 'tolist') else optimized_thetas,
    'description': 'Optimized theta values for CRY gates',
    'target_frequencies': {
        'first_3_qubits': pt_f_co.tolist(),
        'last_4_qubits': pt_c_co.tolist()
    },
    'optimization_result': {
        'success': result.success if 'result' in locals() and hasattr(result, 'success') else False,
        'final_cost': float(result.fun) if 'result' in locals() and hasattr(result, 'fun') else 0.0,
        'iterations': result.nfev if 'result' in locals() and hasattr(result, 'nfev') else 0
    }
}

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
output_file = os.path.join(script_dir, 'optimized_thetas.json')

try:
    with open(output_file, 'w') as f:
        json.dump(theta_dict, f, indent=2)
    
    print("\n" + "="*60)
    print(f"SUCCESS: Optimized thetas saved to:")
    print(f"{os.path.abspath(output_file)}")
    print("="*60)
    
    # Verify the file was written
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file)
        print(f"File size: {file_size} bytes")
        if file_size == 0:
            print("WARNING: File was created but is empty!")
    else:
        print("ERROR: File was not created!")
        
except Exception as e:
    print("\n" + "!"*60)
    print("ERROR: Failed to save optimized thetas:")
    print(str(e))
    print("Current working directory:", os.getcwd())
    print("!"*60 + "\n")