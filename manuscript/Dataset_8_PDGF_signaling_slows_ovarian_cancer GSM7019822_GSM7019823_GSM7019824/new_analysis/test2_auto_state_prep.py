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







merged_circuit = qc1.tensor(qc2)  # or qc1 ^ qc2
print(f"Merged circuit has {merged_circuit.num_qubits} qubits")  # Should be 7

print(qc1)

merged_circuit = qc1 ^ qc2
