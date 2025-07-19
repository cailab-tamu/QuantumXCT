import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import StatePreparation
from qiskit import transpile
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt


pt_f_mo = [0.234577, 0.182927, 0.091105, 0.061693, 0.164634, 0.125897, 0.080703, 0.058465]
pt_c_mo = [0.118443, 0.006768, 0.048223, 0.002538, 0.078257, 0.005076, 0.053723, 0.006768, 0.140017, 0.013536, 0.096024, 0.013536, 0.170474, 0.017766, 0.195854, 0.032995]

# pt_f_co = [0.240181, 0.061934, 0.144260, 0.035498, 0.230363, 0.047583, 0.197130, 0.043051]
# pt_c_co = [0.115983, 0.019802, 0.053748, 0.019095, 0.077793, 0.022631, 0.056577, 0.025460, 0.083451, 0.026167, 0.062235, 0.034653, 0.108911, 0.049505, 0.142150, 0.101839]

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


# merged_circuit = qc1.tensor(qc2)  # or qc1 ^ qc2
merged_circuit = qc1 ^ qc2
print(f"Merged circuit has {merged_circuit.num_qubits} qubits")  # Should be 7

theta = np.pi/4  # Example angle, you can adjust this value

# Add CRY gates from first 3 qubits to last 4 qubits
for i in range(3):  # First 3 qubits (0,1,2)
    for j in range(3, 7):  # Last 4 qubits (3,4,5,6)
        merged_circuit.cry(theta, i, j)

# Add CRY gates from last 4 qubits to first 3 qubits
for i in range(3, 7):  # Last 4 qubits (3,4,5,6)
    for j in range(3):  # First 3 qubits (0,1,2)
        if i != j:  # Avoid self-loops
            merged_circuit.cry(theta, i, j)

print(merged_circuit)

# Create a copy of the circuit for measurement
measured_circuit = merged_circuit.copy()

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
