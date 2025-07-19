import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import StatePreparation
from qiskit import transpile
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt



# pt_f_mo = [0.234577, 0.182927, 0.091105, 0.061693, 0.164634, 0.125897, 0.080703, 0.058465]
# pt_c_mo = [0.118443, 0.006768, 0.048223, 0.002538, 0.078257, 0.005076, 0.053723, 0.006768, 0.140017, 0.013536, 0.096024, 0.013536, 0.170474, 0.017766, 0.195854, 0.032995]

# pt_f_co = [0.240181, 0.061934, 0.144260, 0.035498, 0.230363, 0.047583, 0.197130, 0.043051]
# pt_c_co = [0.115983, 0.019802, 0.053748, 0.019095, 0.077793, 0.022631, 0.056577, 0.025460, 0.083451, 0.026167, 0.062235, 0.034653, 0.108911, 0.049505, 0.142150, 0.101839]


# Your frequency distribution
frequencies = np.array([0.118443, 0.006768, 0.048223, 0.002538, 0.078257, 0.005076, 
                       0.053723, 0.006768, 0.140017, 0.013536, 0.096024, 0.013536, 
                       0.170474, 0.017766, 0.195854, 0.032995])

# Convert frequencies to amplitudes (square root of normalized probabilities)
# First normalize to ensure sum = 1
normalized_probs = frequencies / np.sum(frequencies)
amplitudes = np.sqrt(normalized_probs)

print("Original frequencies:", frequencies)
print("Normalized probabilities:", normalized_probs)
print("Amplitudes:", amplitudes)
print("Sum of probabilities:", np.sum(normalized_probs))
print("Sum of squared amplitudes:", np.sum(amplitudes**2))

# Create quantum circuit
qreg = QuantumRegister(4, 'q')
creg = ClassicalRegister(4, 'c')
qc = QuantumCircuit(qreg, creg)

# Method 1: Using Qiskit's StatePreparation
# This automatically generates the optimal circuit
state_prep = StatePreparation(amplitudes)
qc.append(state_prep, qreg)

# Add measurements
qc.measure(qreg, creg)

print(f"\nQuantum Circuit (depth: {qc.depth()}):")
print(qc)

# Simulate the circuit
simulator = AerSimulator()
compiled_circuit = transpile(qc, simulator)

# Run simulation
job = simulator.run(compiled_circuit, shots=10000)
result = job.result()
counts = result.get_counts()

# Convert counts to frequencies for comparison
measured_frequencies = np.zeros(16)
total_shots = sum(counts.values())

for bitstring, count in counts.items():
    # Convert bitstring to integer (reverse bit order for standard indexing)
    index = int(bitstring, 2)
    measured_frequencies[index] = count / total_shots

print("\nComparison of target vs measured frequencies:")
print("State | Target   | Measured | Difference")
print("-" * 40)
for i in range(16):
    target = normalized_probs[i]
    measured = measured_frequencies[i]
    diff = abs(target - measured)
    print(f"|{i:04b}⟩ | {target:.6f} | {measured:.6f} | {diff:.6f}")

# Calculate fidelity
fidelity = np.sum(np.sqrt(normalized_probs * measured_frequencies))
print(f"\nFidelity: {fidelity:.6f}")

# Plot comparison
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Target distribution
ax1.bar(range(16), normalized_probs, alpha=0.7, color='blue')
ax1.set_title('Target Distribution')
ax1.set_xlabel('Computational Basis State')
ax1.set_ylabel('Probability')
ax1.set_xticks(range(16))
ax1.set_xticklabels([f'{i:04b}' for i in range(16)], rotation=45)

# Measured distribution
ax2.bar(range(16), measured_frequencies, alpha=0.7, color='red')
ax2.set_title('Measured Distribution')
ax2.set_xlabel('Computational Basis State')
ax2.set_ylabel('Probability')
ax2.set_xticks(range(16))
ax2.set_xticklabels([f'{i:04b}' for i in range(16)], rotation=45)

# Overlay comparison
ax3.bar(range(16), normalized_probs, alpha=0.5, color='blue', label='Target')
ax3.bar(range(16), measured_frequencies, alpha=0.5, color='red', label='Measured')
ax3.set_title('Comparison')
ax3.set_xlabel('Computational Basis State')
ax3.set_ylabel('Probability')
ax3.set_xticks(range(16))
ax3.set_xticklabels([f'{i:04b}' for i in range(16)], rotation=45)
ax3.legend()

plt.tight_layout()
plt.show()

# Manual implementation using uniformly controlled rotations
def create_state_preparation_manual(amplitudes):
    """
    Manual implementation of state preparation using uniformly controlled rotations
    """
    qc_manual = QuantumCircuit(4, 4)
    
    # This is a simplified version - full implementation would require
    # more complex uniformly controlled rotation decomposition
    
    # For demonstration, we'll use a different approach:
    # Apply rotations layer by layer
    
    # First, prepare the first qubit based on the sum of amplitudes
    # for states |0xxx⟩ vs |1xxx⟩
    prob_0 = np.sum(amplitudes[0:8]**2)
    prob_1 = np.sum(amplitudes[8:16]**2)
    
    if prob_0 + prob_1 > 0:
        angle = 2 * np.arctan(np.sqrt(prob_1/prob_0)) if prob_0 > 0 else np.pi
        qc_manual.ry(angle, 0)
    
    # This would continue with more complex controlled rotations
    # For now, we'll use the StatePreparation library function
    
    return qc_manual

print("\nCircuit Statistics:")
print(f"Number of qubits: {qc.num_qubits}")
print(f"Circuit depth: {qc.depth()}")
print(f"Number of gates: {len(qc.data)}")

# Circuit without measurements for visualization
qc_no_measure = QuantumCircuit(4)
qc_no_measure.append(StatePreparation(amplitudes), range(4))
print(f"\nState preparation circuit (no measurements):")
print(qc_no_measure)
