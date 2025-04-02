import matplotlib.pyplot as plt
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.library import TwoLocal
import numpy as np

def create_grn_ansatz(ng, cell_type):
    """
    Creates a parameterized quantum circuit representing a Gene Regulatory Network (GRN) ansatz.

    Args:
        ng: The number of genes in the network.
        cell_type: A string identifier for the cell type (e.g., "CT1", "CT2").

    Returns:
        A QuantumCircuit object representing the GRN ansatz.
    """
    cell_type = cell_type.lower()
    ansatz_grn = QuantumCircuit(ng, name=f"{cell_type}_GRN_Ansatz")

    # Gene activation probabilities (RY rotations after Hadamard)
    params_act = [Parameter(f'{cell_type}_act_{i}') for i in range(ng)]
    #params_act2 = [Parameter(f'{cell_type}_act2_{i}') for i in range(ng)]

    for i in range(ng):
        ansatz_grn.h(i)
        ansatz_grn.rz(params_act[i], i)  # Use RZ for activation
        #ansatz_grn.rx(params_act2[i], i)  # Use RZ for activation
    
    # Gene interaction CRX gates
    for i in range(ng):
        for j in range(ng):
            if i != j:
                param_name = f'{cell_type}_grn_{i}_{j}'
                param = Parameter(param_name)
                ansatz_grn.cry(param, i, j)

    return ansatz_grn

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from qiskit.quantum_info import SparsePauliOp

def create_interaction_observable_from_histogram(joint_counts: Counter, num_features: int, min_ones: int = 1, rm_all_ones: bool=False):
    """Creates a SparsePauliOp from joint histogram counts,
        favoring bit strings with odd number of '1's.

    Args:
        joint_counts: A Counter object from create_joint_histogram.
        num_features: The total number of qubits.
        min_ones: The minimum number of '1's required in a bit string
                  for the interaction to be included.
        rm_all_ones: If True, removes the bit string with all '1's.
    Returns:
        A SparsePauliOp observable.
    """
    interaction_strength_list = []

    for bit_string, count in joint_counts.items():
        num_ones = bit_string.count('1')  # Count the number of '1's

        if num_ones == num_features and rm_all_ones:
            continue

        if num_ones >= min_ones:  # Consider only if at least min_ones '1's are present
            nodes = tuple(i for i, bit in enumerate(bit_string) if bit == '1')

            #strength = -float(count)
            strength = float(count)
            #strength = -1.0*(-1.0*count)**(num_ones) 

            pauli_string = ""
            for i in range(num_features):
                if i in nodes:
                    pauli_string += "Z"
                else:
                    pauli_string += "I"

            interaction_strength_list.append((pauli_string, strength))

    interaction_observable = SparsePauliOp.from_list(interaction_strength_list)
    return interaction_observable


def create_interaction_observable_general(interactions, num_features):
    """Creates a SparsePauliOp observable for generalized interactions.

    Args:
        interactions: A dictionary where keys are tuples of node indices 
                     (e.g., (0, 1), (0, 0, 2), (0, 1, 2, 3)) and 
                     values are the corresponding interaction strengths.
        num_features: The total number of qubits.

    Returns:
        A SparsePauliOp observable.
    """
    interaction_strength_list = []
    for nodes, strength in interactions.items():
        pauli_string = ""
        for i in range(num_features):
            if i in nodes:  # Check if the current qubit is in the interaction
                pauli_string += "Z"
            else:
                pauli_string += "I"
        interaction_strength_list.append((pauli_string, strength))

    interaction_observable = SparsePauliOp.from_list(interaction_strength_list)
    return interaction_observable


import matplotlib.pyplot as plt
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
#from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler  # If using IBM Runtime
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.visualization import plot_histogram

def evaluate_and_plot_ansatz(ansatz, params, shots=1024, title="Quantum Sampler Results"):
    """Evaluates a quantum ansatz, plots results and circuit, and prints counts."""

    try:
        sampler = StatevectorSampler()
        bound_circuit = ansatz.copy()
        bound_circuit.assign_parameters(params, inplace=True)
        bound_circuit.measure_all()

        job = sampler.run([bound_circuit], shots=shots)
        pub_result = job.result()[0]
        data_pub = pub_result.data
        counts = data_pub.meas.get_counts()

        print(f"The counts are: {counts}")

        # Plot histogram:
        plot_histogram(counts, bar_labels=True, title=title).show()

        # Matplotlib customization:
        sorted_counts = dict(sorted(counts.items()))
        x_labels = list(sorted_counts.keys())
        y_values = list(sorted_counts.values())

        plt.figure(figsize=(12, 8))
        plt.bar(x_labels, y_values)
        plt.xlabel("Measurement Outcomes", fontsize=16)
        plt.ylabel("Counts", fontsize=16)
        plt.title(title, fontsize=18)
        plt.xticks(rotation=45, ha='right', fontsize=14)
        plt.tight_layout()
        plt.show()

        return counts, bound_circuit

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def create_parameter_dictionaries(combined_qc, ct1_percentages):
    """Creates static and variable parameter dictionaries."""
    # Get Hadamard parameters
    # params_ct = [param for param in combined_qc.parameters if 'ct1_' in param.name or 'ct2_' in param.name
    #                                                         and 'grn' not in param.name 
    #                                                         and 'lr' not in param.name 
    #                                                         and 'act2' not in param.name]
    params_ct = [param for param in combined_qc.parameters if '_act_' in param.name]

    static_params = {}
    for i, val in enumerate(ct1_percentages):
        static_params[params_ct[i]] = val

    variable_params = [param for param in combined_qc.parameters if param not in static_params]
    return static_params, variable_params

from qiskit.quantum_info import Statevector

def cost_func_vqe(params, combined_qc, hamiltonian, estimator):  # combined_qc here
    """Cost function for VQE"""
    bound_qc = combined_qc.assign_parameters(params)  # Assign parameters INSIDE cost_func_vqe
    statevector = Statevector(bound_qc)  # Use bound_qc
    statevector_array = statevector.data
    hamiltonian_matrix = hamiltonian.to_matrix()
    energy = np.real(statevector_array.conjugate() @ hamiltonian_matrix @ statevector_array)
    return energy

def cost_func_wrapper(variable_values, all_params, combined_qc, interaction_observable, estimator, variable_params):
    for i, param in enumerate(variable_params):
        all_params[param] = variable_values[i]
    return cost_func_vqe(all_params, combined_qc, interaction_observable, estimator) # Pass combined_qc

# Create the static and variable parameter dictionaries directly from the circuit.
def create_parameter_dictionaries_from_circuit(circuit):
    """Creates static and variable parameter dictionaries directly from the circuit."""
    static_params = {param: None for param in circuit.parameters if 'lr_' not in param.name}
    variable_params = {param.name: param for param in circuit.parameters if 'lr_' in param.name}
    return static_params, variable_params

from qiskit.circuit import QuantumCircuit, Parameter

def create_circuit_lr2(ansatz_grn_ct1, ansatz_grn_ct2, cell_type1='ct1', cell_type2='ct2', interactions=None):
    """Concatenates two circuits and includes interactions using CRY rotations with parameterized angles."""
    ng_ct1 = ansatz_grn_ct1.num_qubits
    ng_ct2 = ansatz_grn_ct2.num_qubits
    cell_type1 = cell_type1.lower()
    cell_type2 = cell_type2.lower()
    num_features = ng_ct1 + ng_ct2
    ccgrn_circuit = QuantumCircuit(num_features, name=f"CC_GRN_Ansatz")

    # Gene activation probabilities (RZ rotations after Hadamard) for ct1
    params_act_ct1 = [Parameter(f'{cell_type1}_act_{i}') for i in range(ng_ct1)]
    #params_act_ct12 = [Parameter(f'{cell_type1}_act2_{i}') for i in range(ng_ct1)]
    for i in range(ng_ct1):
        ccgrn_circuit.h(i)
        ccgrn_circuit.rz(params_act_ct1[i], i)
        #ccgrn_circuit.rx(params_act_ct12[i], i)

    # Gene interaction CRX gates for ct1
    for i in range(ng_ct1):
        for j in range(ng_ct1):
            if i != j:
                param_name = f'{cell_type1}_grn_{i}_{j}'
                param = Parameter(param_name)
                ccgrn_circuit.cry(param, i, j)
                #ccgrn_circuit.crx(param, i, j)


    # Gene activation probabilities (RZ rotations after Hadamard) for ct2
    params_act_ct2 = [Parameter(f'{cell_type2}_act_{i}') for i in range(ng_ct2)]
    #params_act_ct22 = [Parameter(f'{cell_type2}_act2_{i}') for i in range(ng_ct2)]
    for i, j in enumerate(range(ng_ct1, num_features)):
        ccgrn_circuit.h(j)
        ccgrn_circuit.rz(params_act_ct2[i], j)  # Corrected indexing here
        #ccgrn_circuit.rx(params_act_ct22[i], j)  # Corrected indexing here

    # Gene interaction CRX gates for ct2
    for i, q1 in enumerate(range(ng_ct1, num_features)):
        for j, q2 in enumerate(range(ng_ct1, num_features)):
            if q1 != q2:
                param_name = f'{cell_type2}_grn_{i}_{j}'
                param = Parameter(param_name)
                ccgrn_circuit.cry(param, q1, q2)
                #ccgrn_circuit.crx(param, q1, q2)

    # Add interactions if provided
    if interactions:
        for (q1, q2) in interactions.keys():
            if not (0 <= q1 < num_features and 0 <= q2 < num_features):
                raise ValueError("Qubit indices in interactions are out of range.")
            param_name = f"lr_{q1}_{q2}"
            angle_param = Parameter(param_name)
            #ccgrn_circuit.cry(angle_param, q1, q2)
            ccgrn_circuit.crx(angle_param, q1, q2)


    return ccgrn_circuit