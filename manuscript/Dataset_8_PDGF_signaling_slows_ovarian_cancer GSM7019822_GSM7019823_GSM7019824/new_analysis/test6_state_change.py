from qiskit import QuantumCircuit
from qiskit.circuit.library import Initialize
import numpy as np

# psi_1 and psi_2 are *normalized* numpy arrays of 2**n amplitudes
psi_1 = np.array([0.25, 0.25, 0.25, 0.25])
psi_2 = np.array([0.1, 0.2, 0.3, 0.4])

n = int(np.log2(len(psi_1)))

qc = QuantumCircuit(n)

# Un-preparation: psi_1^dagger (adjoint of the gate that prepares psi_1 from |0>)
init1 = Initialize(psi_1)
init1_circ = init1.gates_to_uncompute()   # The inverse circuit

# Preparation: prepare psi_2 from |0>
init2 = Initialize(psi_2)

# Compose: first uncompute psi_1, then compute psi_2
qc.append(init1_circ, range(n))
qc.append(init2, range(n))


