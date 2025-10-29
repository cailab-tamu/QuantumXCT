import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector, DensityMatrix
from qiskit.circuit.library import Initialize
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.visualization import plot_histogram
from scipy.special import rel_entr
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import random
import time

# --- Modified create_initial_circuit function ---
def create_initial_circuit2(sparse_amplitude_dict):
    """
    Creates a quantum circuit initialized to a given state vector.
    This function now directly accepts a sparse dictionary, infers num_qubits,
    pads non-specified states with zeros, and L2-normalizes the state vector internally.
    Args:
        sparse_amplitude_dict (dict): A dictionary where keys are bitstrings
                                      (e.g., '000', '101') and values are
                                      relative amplitudes (can be integers or floats).
    Returns:
        QuantumCircuit: The circuit initialized to the specified state.
    Raises:
        ValueError: If the dictionary is empty or bitstring lengths are inconsistent,
                    or if the resulting state vector cannot be normalized.
    """
    if not sparse_amplitude_dict:
        raise ValueError("Input dictionary cannot be empty.")

    # 1. Infer num_qubits from the bitstring length
    sample_bitstring = next(iter(sparse_amplitude_dict))
    num_qubits = len(sample_bitstring)

    # Optional: Verify consistency of bitstring lengths (good practice)
    for bitstring in sparse_amplitude_dict.keys():
        if len(bitstring) != num_qubits:
            raise ValueError("All bitstrings in the dictionary must have the same length to infer num_qubits.")

    # 2. Create a full state vector template initialized to zeros
    vector_length = 2**num_qubits
    full_state_vector = np.zeros(vector_length, dtype=float)

    # 3. Populate the template with provided values
    for bitstring, value in sparse_amplitude_dict.items():
        index = int(bitstring, 2) # Convert binary string to integer index
        full_state_vector[index] = float(value) # Ensure values are floats

    # 4. Normalize the full state vector (L2 norm)
    norm_val = np.linalg.norm(full_state_vector)

    if norm_val == 0:
        # If all values are zero, the normalized vector is also all zeros.
        # This represents the |0...0> state.
        state_vec_probs = full_state_vector
    else:
        state_vec_probs = full_state_vector / norm_val
    
    # Verify L2-normalization for good measure (should always pass if norm_val != 0)
    if not np.isclose(np.linalg.norm(state_vec_probs), 1.0) and norm_val != 0:
         raise ValueError("Internal error: State vector did not normalize correctly.")

    # 5. Create the Qiskit Quantum Circuit
    qr = QuantumRegister(num_qubits, name='q')
    circuit = QuantumCircuit(qr)
    initial_state_instruction = Initialize(state_vec_probs)
    circuit.append(initial_state_instruction, qr)
    
    return circuit

def concatenate_circuits_with_separate_measurements(circ1: QuantumCircuit, circ2: QuantumCircuit) -> QuantumCircuit:
    """
    Concatenates two QuantumCircuit objects onto disjoint sets of qubits
    within a larger circuit and adds separate classical registers for measurement
    of each original circuit's qubits.

    Args:
        circ1 (QuantumCircuit): The first quantum circuit.
        circ2 (QuantumCircuit): The second quantum circuit.

    Returns:
        QuantumCircuit: A new circuit combining circ1 and circ2 on separate
                        qubits, with two distinct classical registers for measurements.
    """
    ng_circ1 = circ1.num_qubits
    ng_circ2 = circ2.num_qubits
    num_total_qubits = ng_circ1 + ng_circ2

    qr_all = QuantumRegister(num_total_qubits, name='q')
    cr_measure1 = ClassicalRegister(ng_circ1, name='c_measure1')
    cr_measure2 = ClassicalRegister(ng_circ2, name='c_measure2')

    circ_all = QuantumCircuit(qr_all, cr_measure1, cr_measure2)

    # Compose circ1 onto the first set of qubits
    circ_all.compose(circ1, qubits=range(ng_circ1), inplace=True)

    # Compose circ2 onto the next set of qubits
    circ_all.compose(circ2, qubits=range(ng_circ1, num_total_qubits), inplace=True)

    return circ_all


def add_cnots_and_measurements_to_circuit(
    base_circuit: QuantumCircuit,
    circ1_num_qubits: int,
    global_cnot_configurations: list[tuple[int, int]]
) -> QuantumCircuit:
    """
    Applies a specified list of CNOT gates (using global qubit indices)
    and then adds measurements to the circuit.

    Args:
        base_circuit (QuantumCircuit): The circuit already containing the two
                                        chunks composed on disjoint qubits.
                                        This circuit should NOT have measurements yet.
        circ1_num_qubits (int): The number of qubits in the first chunk.
                                This is used to determine the classical register split.
        global_cnot_configurations (list[tuple[int, int]]): A list of tuples, where each tuple
                                                      (global_control_idx, global_target_idx)
                                                      specifies a CNOT gate using global qubit indices.

    Returns:
        QuantumCircuit: A new circuit with the specified CNOTs and measurements added.
    """
    circuit_with_cnots = base_circuit.copy()

    qr_all = circuit_with_cnots.qregs[0]
    cr_measure1 = circuit_with_cnots.cregs[0]
    cr_measure2 = circuit_with_cnots.cregs[1]

    for control_q, target_q in global_cnot_configurations:
        # Add checks to ensure indices are valid within the combined circuit
        if not (0 <= control_q < circuit_with_cnots.num_qubits and
                0 <= target_q < circuit_with_cnots.num_qubits and
                control_q != target_q):
            raise ValueError(f"Invalid CNOT indices: ({control_q}, {target_q}). Qubits must be valid and distinct.")

        circuit_with_cnots.cx(qr_all[control_q], qr_all[target_q])
        #circuit_with_cnots.cy(qr_all[control_q], qr_all[target_q])

    # Add measurements after all CNOTs are applied
    circuit_with_cnots.measure(qr_all[0:circ1_num_qubits], cr_measure1)
    circuit_with_cnots.measure(qr_all[circ1_num_qubits:circuit_with_cnots.num_qubits], cr_measure2)

    return circuit_with_cnots

def add_crx_gates_and_measurements_to_circuit(
    base_circuit: QuantumCircuit,
    circ1_num_qubits: int,
    crx_configurations: list[tuple[int, int]], # List of (control, target) global indices
    angles: list[float] # List of angles corresponding to each CRX
) -> QuantumCircuit:
    """
    Applies a specified list of CRX gates (controlled-RX) with given angles
    and then adds measurements to the circuit.

    Args:
        base_circuit (QuantumCircuit): The circuit already containing the two
                                        chunks composed on disjoint qubits.
                                        This circuit should NOT have measurements yet.
        circ1_num_qubits (int): The number of qubits in the first chunk.
                                This is used to determine the classical register split.
        crx_configurations (list[tuple[int, int]]): A list of (control_q, target_q) global qubit
                                                     indices defining where CRX gates will be placed.
        angles (list[float]): A list of rotation angles for each CRX gate,
                              corresponding to the order in `crx_configurations`.

    Returns:
        QuantumCircuit: A new circuit with the specified CRX gates and measurements added.
    """
    circuit_with_crx = base_circuit.copy()
    qr_all = circuit_with_crx.qregs[0]
    cr_measure1 = circuit_with_crx.cregs[0]
    cr_measure2 = circuit_with_crx.cregs[1]

    if len(crx_configurations) != len(angles):
        raise ValueError("Number of CRX configurations must match the number of angles.")

    for i, (control_q, target_q) in enumerate(crx_configurations):
        # Add checks for valid indices
        if not (0 <= control_q < circuit_with_crx.num_qubits and
                0 <= target_q < circuit_with_crx.num_qubits and
                control_q != target_q):
            raise ValueError(f"Invalid CRX indices: ({control_q}, {target_q}). Qubits must be valid and distinct.")
        
        #circuit_with_crx.append(CRXGate(angles[i]), [qr_all[control_q], qr_all[target_q]])
        circuit_with_crx.crx(angles[i], qr_all[control_q], qr_all[target_q]) 

    # Add measurements after all CRX gates are applied
    circuit_with_crx.measure(qr_all[0:circ1_num_qubits], cr_measure1)
    circuit_with_crx.measure(qr_all[circ1_num_qubits:circuit_with_crx.num_qubits], cr_measure2)

    return circuit_with_crx


def get_probability_distribution(counts: dict, num_bits: int, total_shots: int) -> dict:
    """
    Converts a counts dictionary (e.g., {'00': 500, '11': 500}) into
    a probability distribution (e.g., {'00': 0.5, '11': 0.5}).
    Ensures all 2^num_bits outcomes are represented, even if count is 0.
    """
    prob_dist = {}
    for i in range(2**num_bits):
        bitstring = bin(i)[2:].zfill(num_bits)
        prob_dist[bitstring] = counts.get(bitstring, 0) / total_shots
    return prob_dist


def calculate_kl_divergence(p_dist: dict, q_dist: dict, epsilon=1e-9) -> float:
    """
    Calculates the Kullback-Leibler divergence D_KL(P || Q).
    P is the observed distribution, Q is the target distribution.
    Args:
        p_dist (dict): Observed probability distribution (e.g., from simulation).
        q_dist (dict): Target probability distribution.
        epsilon (float): A small value used to handle zero probabilities,
                         to avoid log(0) and division by zero.
    Returns:
        float: The KL divergence.
    """
    kl_div = 0.0

    # Get all possible outcomes from the union of keys to ensure comprehensive check.
    all_keys = sorted(list(set(p_dist.keys()).union(set(q_dist.keys()))))

    for key in all_keys:
        p_val = p_dist.get(key, 0.0) # Observed probability for this outcome
        q_val = q_dist.get(key, 0.0) # Target probability for this outcome

        if p_val > 0: # Only sum if the observed probability is non-zero
            if q_val == 0:
                kl_div += p_val * np.log(p_val / epsilon)
            else:
                kl_div += p_val * np.log(p_val / q_val)

    return kl_div

# --- NEW HELPER FUNCTION ---
def _process_target_state_input(target_input):
    """Converts a dictionary or array to a normalized NumPy array."""
    if isinstance(target_input, dict):
        if not target_input: raise ValueError("Target dictionary cannot be empty.")
        sample_bitstring = next(iter(target_input))
        n_qubits = len(sample_bitstring)
        vector_len = 2**n_qubits
        full_vec = np.zeros(vector_len, dtype=float)
        for bs, val in target_input.items():
            idx = int(bs, 2)
            full_vec[idx] = float(val)
        norm = np.linalg.norm(full_vec)
        normalized_vec = full_vec / norm if norm != 0 else full_vec
        return normalized_vec, n_qubits
    elif isinstance(target_input, (list, np.ndarray)):
        processed_vec = np.array(target_input, dtype=float)
        n_qubits = int(np.log2(len(processed_vec)))
        norm = np.linalg.norm(processed_vec)
        if not np.isclose(norm, 1.0) and norm != 0:
            processed_vec = processed_vec / norm
        return processed_vec, n_qubits
    else:
        raise TypeError("Target state must be a dictionary, list, or numpy array.")



def score_circuit_kl_divergences(
    circuit_to_evaluate: QuantumCircuit,
    state_vec_probs_target1: list or np.ndarray,
    state_vec_probs_target2: list or np.ndarray,
    nshots: int = 1000
):
    """
    Evaluates a Qiskit circuit by simulating it and calculates KL divergences
    against provided target probability distributions derived from specific
    state vectors.

    Args:
        circuit_to_evaluate (QuantumCircuit): The Qiskit QuantumCircuit to simulate.
                                              This circuit should already have the
                                              desired CNOT configurations and measurements
                                              applied, and classical registers named
                                              'c_measure1' and 'c_measure2'.
        state_vec_probs_target1 (list or np.array): The amplitudes of the target state vector
                                                     for the first classical register (c_measure1).
        state_vec_probs_target2 (list or np.array): The amplitudes of the target state vector
                                                     for the second classical register (c_measure2).
        nshots (int): The number of shots for the simulation. Defaults to 1000.

    Returns:
        tuple: A tuple containing (kl_div1, kl_div2), which are the KL divergence
               values for c_measure1 and c_measure2 respectively.
               Returns (None, None) if an error occurs during simulation or result retrieval.
    """
    # # Determine number of qubits for target distributions from the length of state vectors
    # num_qubits_target1 = int(np.log2(len(state_vec_probs_target1)))
    # num_qubits_target2 = int(np.log2(len(state_vec_probs_target2)))

    # # 2. Define target states (convert amplitudes to probability distributions)
    # prob_dist_target1 = {bin(i)[2:].zfill(num_qubits_target1): np.abs(state_vec_probs_target1[i])**2
    #                      for i in range(2**num_qubits_target1)}

    # prob_dist_target2 = {bin(i)[2:].zfill(num_qubits_target2): np.abs(state_vec_probs_target2[i])**2
    #                      for i in range(2**num_qubits_target2)}
    
    # --- FIX STARTS HERE ---
    # Process the input target states to ensure they are full, normalized NumPy arrays
    processed_state_vec_probs_target1, num_qubits_target1 = _process_target_state_input(state_vec_probs_target1)
    processed_state_vec_probs_target2, num_qubits_target2 = _process_target_state_input(state_vec_probs_target2)

    # 2. Define target states (convert amplitudes to probability distributions)
    # These lines now correctly operate on the processed (dense, numerically indexed) arrays
    prob_dist_target1 = {bin(i)[2:].zfill(num_qubits_target1): np.abs(processed_state_vec_probs_target1[i])**2
                         for i in range(2**num_qubits_target1)}

    prob_dist_target2 = {bin(i)[2:].zfill(num_qubits_target2): np.abs(processed_state_vec_probs_target2[i])**2
                         for i in range(2**num_qubits_target2)}
    # --- FIX ENDS HERE ---

    # 5. Simulate the circuit
    backend = AerSimulator()
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    qc_comp = pm.run(circuit_to_evaluate)

    sampler = Sampler(mode=backend)
    job = sampler.run([qc_comp], shots=nshots)

    # Access results using the classical register names
    try:
        result = job.result()[0]
        counts_measure1 = result.data.c_measure1.get_counts()
        counts_measure2 = result.data.c_measure2.get_counts()
    except AttributeError as e:
        print(f"Error accessing classical register counts: {e}")
        print("Please ensure your circuit has classical registers named 'c_measure1' and 'c_measure2'.")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred during simulation or result retrieval: {e}")
        return None, None

    # 6. Convert raw counts to probability distributions
    num_bits_cr1 = circuit_to_evaluate.cregs[0].size if circuit_to_evaluate.cregs else 0
    num_bits_cr2 = circuit_to_evaluate.cregs[1].size if len(circuit_to_evaluate.cregs) > 1 else 0

    prob_dist_sim1 = get_probability_distribution(counts_measure1, num_bits_cr1, nshots)
    prob_dist_sim2 = get_probability_distribution(counts_measure2, num_bits_cr2, nshots)

    # 7. Calculate KL Divergence
    kl_div1 = calculate_kl_divergence(prob_dist_sim1, prob_dist_target1)
    kl_div2 = calculate_kl_divergence(prob_dist_sim2, prob_dist_target2)

    return kl_div1, kl_div2

from typing import Tuple
def plot_measurement_histograms(circuit: QuantumCircuit, nshots: int = 1000, title_prefix: str = "", figure_save_name: str = None, figsize: Tuple[int, int] = (12, 5)
):
    """
    Simulates the given circuit and plots histograms for its classical registers
    (c_measure1 and c_measure2) side-by-side. The plots are displayed interactively
    and can optionally be saved to a file.

    Args:
        circuit (QuantumCircuit): The Qiskit QuantumCircuit to simulate and plot.
                                  It must have classical registers named 'c_measure1' and 'c_measure2'.
        nshots (int): The number of shots for the simulation. Defaults to 1000.
        title_prefix (str): A prefix to add to the overall figure title (e.g., "Final Circuit").
        figure_save_name (str, optional): If provided, the figure will be saved to this filename.
                                          Defaults to None (figure only displayed interactively).
    """
    print(f"\n--- Simulating and Plotting Histograms for: {title_prefix} ---")

    # 1. Simulate the circuit
    backend = AerSimulator()
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    qc_comp = pm.run(circuit)

    sampler = Sampler(mode=backend)
    job = sampler.run([qc_comp], shots=nshots)

    # 2. Access results and plot histograms
    try:
        result = job.result()[0]
        
        counts_measure1 = None
        counts_measure2 = None

        # Check if classical registers exist and get counts
        if 'c_measure1' in [creg.name for creg in circuit.cregs]:
            counts_measure1 = result.data.c_measure1.get_counts()
            print(f"Counts for c_measure1: {counts_measure1}")
        else:
            print("Warning: Classical register 'c_measure1' not found in circuit. Skipping histogram for c_measure1.")

        if 'c_measure2' in [creg.name for creg in circuit.cregs]:
            counts_measure2 = result.data.c_measure2.get_counts()
            print(f"Counts for c_measure2: {counts_measure2}")
        else:
            print("Warning: Classical register 'c_measure2' not found in circuit. Skipping histogram for c_measure2.")

        # Create a figure with two subplots
        if counts_measure1 is not None or counts_measure2 is not None:
            fig, axes = plt.subplots(1, 2, figsize=figsize) # 1 row, 2 columns
            fig.suptitle(f"{title_prefix} - Measurement Counts ({nshots} shots)", fontsize=16)

            if counts_measure1 is not None:
                plot_histogram(counts_measure1, ax=axes[0], title="c_measure1")
                axes[0].set_title("c_measure1") # Manually set the title
            else:
                axes[0].set_title("c_measure1 (Not Found)")
                axes[0].text(0.5, 0.5, "No data", horizontalalignment='center', verticalalignment='center', transform=axes[0].transAxes)

            if counts_measure2 is not None:
                plot_histogram(counts_measure2, ax=axes[1], title="c_measure2")
                axes[1].set_title("c_measure2") # Manually set the title
            else:
                axes[1].set_title("c_measure2 (Not Found)")
                axes[1].text(0.5, 0.5, "No data", horizontalalignment='center', verticalalignment='center', transform=axes[1].transAxes)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
            plt.show() # Display the figure

            if figure_save_name:
                try:
                    fig.savefig(figure_save_name)
                    print(f"Histogram figure saved to {figure_save_name}")
                except Exception as save_e:
                    print(f"Error saving figure to {figure_save_name}: {save_e}")
                finally:
                    plt.close(fig) # Close the figure after showing/saving to free memory
        else:
            print("No classical register data available to plot histograms.")

    except AttributeError as e:
        print(f"Error accessing classical register counts: {e}")
        print("Please ensure your circuit has classical registers named 'c_measure1' and 'c_measure2' and measurements are applied.")
    except Exception as e:
        print(f"An unexpected error occurred during simulation or plotting: {e}")
    
    return counts_measure1, counts_measure2


def create_cnot_pairs_from_locations(
    locations_A: list[int],
    locations_B: list[int]
) -> set[tuple[int, int]]:
    """
    Generates a set of unique CNOT pairs from two lists of qubit locations.
    It creates CNOTs in both directions (A -> B and B -> A).
    
    Args:
        locations_A (list[int]): A list of active qubit indices for one circuit.
        locations_B (list[int]): A list of active qubit indices for the other circuit.
        
    Returns:
        set[tuple[int, int]]: A set of unique (control, target) CNOT pairs.
    """
    cnot_pairs = set()

    # Generate CNOTs from A to B
    for control_q in locations_A:
        for target_q in locations_B:
            if control_q != target_q:
                cnot_pairs.add((control_q, target_q))

    # Generate CNOTs from B to A
    for control_q in locations_B:
        for target_q in locations_A:
            if control_q != target_q:
                cnot_pairs.add((control_q, target_q))
                
    return cnot_pairs


def optimize_crx_angles(
    circ1: QuantumCircuit,
    circ2: QuantumCircuit,
    state_vec_probs_target1: list or np.ndarray,
    state_vec_probs_target2: list or np.ndarray,
    cnot_topology: list[tuple[int, int]], # The fixed CNOT structure (control, target)
    nshots: int = 1000,
    etol: float = 1e-6,
    opt_method: str = 'L-BFGS-B', # Optimization method to use "L-BFGS-B", "COBYLA", etc.
    initial_angle_value: float = np.pi # Default initial angle for all CRX gates
):
    """
    Optimizes the rotation angles of CRX gates placed according to a given CNOT topology
    to minimize the combined KL divergence.

    Args:
        circ1 (QuantumCircuit): The first quantum circuit.
        circ2 (QuantumCircuit): The second quantum circuit.
        state_vec_probs_target1 (list or np.array): Target state vector for c_measure1.
        state_vec_probs_target2 (list or np.array): Target state vector for c_measure2.
        cnot_topology (list[tuple[int, int]]): A list of (control_q, target_q) global qubit
                                                indices defining where CRX gates will be placed.
        nshots (int): Number of shots for simulation.
        initial_angle_value (float): The initial angle value for all CRX gates. Defaults to pi.

    Returns:
        tuple: A tuple containing (optimized_angles, min_kl_sum_optimized, optimization_history).
               - optimized_angles: A list of the optimized CRX angles.
               - min_kl_sum_optimized: The minimum combined KL divergence found.
               - optimization_history: A list of KL sums at each iteration of the optimization.
    """
    ng_circ1 = circ1.num_qubits
    num_crx_gates = len(cnot_topology)

    if num_crx_gates == 0:
        print("No CNOT topology provided for CRX optimization. Returning baseline KL sum.")
        base_combined_circuit = concatenate_circuits_with_separate_measurements(circ1, circ2)
        base_circuit_with_measurements = add_cnots_and_measurements_to_circuit(
            base_combined_circuit, ng_circ1, [] # No CNOTs
        )
        kl_div1, kl_div2 = score_circuit_kl_divergences(
            base_circuit_with_measurements,
            state_vec_probs_target1,
            state_vec_probs_target2,
            nshots
        )
        return [], (kl_div1 + kl_div2 if kl_div1 is not None and kl_div2 is not None else float('inf')), []


    print(f"\n--- Starting CRX Angle Optimization ({num_crx_gates} CRX gates) ---")

    # Initial guess for angles (all pi, as CRX(pi) is CNOT)
    initial_angles = np.array([initial_angle_value] * num_crx_gates)

    # List to store the KL sum at each iteration
    optimization_history = []

    # Define the objective function for the optimizer
    def objective_function(angles_array):
        # Ensure angles_array is a list for add_crx_gates_and_measurements_to_circuit
        angles_list = angles_array.tolist()

        # Build the circuit with CRX gates and current angles
        base_combined_circuit = concatenate_circuits_with_separate_measurements(circ1, circ2)
        circuit_with_crx_and_measurements = add_crx_gates_and_measurements_to_circuit(
            base_combined_circuit, ng_circ1, cnot_topology, angles_list
        )

        # Score the circuit
        kl_div1, kl_div2 = score_circuit_kl_divergences(
            circuit_with_crx_and_measurements,
            state_vec_probs_target1,
            state_vec_probs_target2,
            nshots
        )

        # Return the sum of KL divergences (or a large value if an error occurred)
        current_kl_sum = kl_div1 + kl_div2 if kl_div1 is not None and kl_div2 is not None else float('inf')
        
        # Append to history
        optimization_history.append(current_kl_sum)
        
        return current_kl_sum

    # Define bounds for the angles (e.g., 0 to 2*pi)
    bounds = [(0, np.pi)] * num_crx_gates

    # Perform the minimization
    start_time = time.time()
    result = minimize(objective_function, 
                      initial_angles, 
                      method = opt_method, 
                      bounds = bounds, 
                      tol=etol, 
                      options = {'disp': True, 'maxiter': 500})
    end_time = time.time()

    optimized_angles = result.x.tolist()
    min_kl_sum_optimized = result.fun

    print(f"\nOptimization Results:")
    print(f"  Success: {result.success}")
    print(f"  Message: {result.message}")
    print(f"  Optimized Angles: {[f'{angle:.4f}' for angle in optimized_angles]}")
    #print(f"  Minimum Combined KL Divergence: {min_kl_sum_optimized:.6f}")
    print(f"CRX Angle Optimization took: {end_time - start_time:.2f} seconds")

    return optimized_angles, min_kl_sum_optimized, optimization_history


def find_best_cnot_sequence_brute_force(
    circ1: QuantumCircuit,
    circ2: QuantumCircuit,
    state_vec_probs_target1: list or np.ndarray,
    state_vec_probs_target2: list or np.ndarray,
    max_cnot_depth: int = 1, # Limit the depth for practical reasons
    nshots: int = 1000,
    kl_tol: float = 0.01
):
    """
    Performs a brute-force search to find the optimal sequence of CNOT gates
    between two quantum circuit "chunks" (circ1 and circ2) up to a specified depth.
    It evaluates all possible permutations of CNOTs and returns the sequence
    that yields the lowest combined KL divergence.

    Args:
        circ1 (QuantumCircuit): The first quantum circuit (e.g., circ_bell).
        circ2 (QuantumCircuit): The second quantum circuit (e.g., circ_ghz_ish).
        state_vec_probs_target1 (list or np.array): The amplitudes of the target state vector
                                                     for the first classical register (c_measure1).
        state_vec_probs_target2 (list or np.array): The amplitudes of the target state vector
                                                     for the second classical register (c_measure2).
        max_cnot_depth (int): The maximum number of CNOT gates to include in a sequence.
                              Be cautious: computational cost grows exponentially with this value.
        nshots (int): The number of shots for the simulation. Defaults to 1000.

    Returns:
        tuple: A tuple containing (best_cnot_sequence, min_kl_sum).
               - best_cnot_sequence: A list of (global_control_idx, global_target_idx) tuples
                                     representing the optimal sequence of CNOTs found.
                                     An empty list if no CNOTs improve the baseline.
               - min_kl_sum: The minimum combined KL divergence found.
    """
    # Define a global tolerance for KL divergence stopping criteria
    ng_circ1 = circ1.num_qubits
    ng_circ2 = circ2.num_qubits
    num_total_qubits = ng_circ1 + ng_circ2

    print(f"\n--- Starting Brute-Force CNOT Sequence Optimization (Total Qubits: {num_total_qubits}, Max Depth: {max_cnot_depth}) ---")
    print(f"Number of qubits in chunk 1: {ng_circ1}")
    print(f"Number of qubits in chunk 2: {ng_circ2}")

    # 1. Evaluate baseline (no CNOTs)
    base_combined_circuit = concatenate_circuits_with_separate_measurements(circ1, circ2)
    base_circuit_with_measurements = add_cnots_and_measurements_to_circuit(
        base_combined_circuit, ng_circ1, []
    )
    kl_div1_no_cnot, kl_div2_no_cnot = score_circuit_kl_divergences(
        base_circuit_with_measurements,
        state_vec_probs_target1,
        state_vec_probs_target2,
        nshots
    )
    initial_kl_sum = kl_div1_no_cnot + kl_div2_no_cnot if kl_div1_no_cnot is not None and kl_div2_no_cnot is not None else float('inf')

    min_kl_sum = initial_kl_sum
    best_cnot_sequence = [] # List of (global_control, global_target) tuples

    print(f"\nBaseline (No CNOTs) KL Sum: {initial_kl_sum:.6f}")
    print(f"Initial best KL sum: {min_kl_sum:.6f} (from baseline)")

    # Generate all possible single CNOTs between the two chunks (global indices)
    all_possible_single_cnots = []
    for control_q1_idx in range(ng_circ1):
        for target_q2_idx in range(ng_circ2):
            # CNOT from chunk1 to chunk2
            all_possible_single_cnots.append((control_q1_idx, ng_circ1 + target_q2_idx))
            # CNOT from chunk2 to chunk1
            all_possible_single_cnots.append((ng_circ1 + target_q2_idx, control_q1_idx))

    print(f"\n Number of possible single cnots {len(all_possible_single_cnots)}")
    # Brute-force search for the best sequence of CNOTs up to max_cnot_depth
    start_time = time.time()
    for num_cnots in range(1, max_cnot_depth + 1):
        print(f"\n--- Testing combinations with {num_cnots} CNOTs ---")
        # Use permutations to account for order, as CNOT order matters
        for cnot_combination in itertools.permutations(all_possible_single_cnots, num_cnots):
            temp_cnot_sequence = list(cnot_combination)

            # Create circuit with current CNOT sequence
            temp_base_circuit = concatenate_circuits_with_separate_measurements(circ1, circ2)
            temp_circuit_with_cnots = add_cnots_and_measurements_to_circuit(
                temp_base_circuit, ng_circ1, temp_cnot_sequence
            )

            kl_div1, kl_div2 = score_circuit_kl_divergences(
                temp_circuit_with_cnots,
                state_vec_probs_target1,
                state_vec_probs_target2,
                nshots
            )

            if kl_div1 is not None and kl_div2 is not None:
                current_kl_sum = kl_div1 + kl_div2
                # print(f"  Testing CNOT sequence {temp_cnot_sequence}: KL Sum {current_kl_sum:.6f}") # For debugging brute force

                if current_kl_sum < min_kl_sum:
                    min_kl_sum = current_kl_sum
                    best_cnot_sequence = temp_cnot_sequence
                    print(f"  --> New best sequence found: {best_cnot_sequence} with KL Sum: {min_kl_sum:.6f}")
                
                # Check for early stopping criterion
                if min_kl_sum < kl_tol:
                    print(f"  KL sum ({min_kl_sum:.6f}) below tolerance {kl_tol}. Stopping early.")
                    search_complete_early = True
                    break # Exit inner loop

    end_time = time.time()
    print(f"Brute-Force CNOT search (max_depth={max_cnot_depth}) took: {end_time - start_time:.2f} seconds")
    
    # If no CNOT sequence improved over the baseline, return the baseline results
    if min_kl_sum >= initial_kl_sum:
        return [], initial_kl_sum
    else:
        return best_cnot_sequence, min_kl_sum

# ==============================================================================
# --- MAIN SEARCH ALGORITHMS (REFACTORED) ---
# ==============================================================================
def _run_single_greedy_search_from_start(
    circ1, circ2, state_vec_probs_target1, state_vec_probs_target2,
    all_possible_single_cnots, starting_cnot, min_cnot_depth, nshots, kl_tol, ratio_kl_tol,
    initial_kl_for_path, # <-- NEW: Accepts pre-calculated KL score
    max_cnot_depth=30    # <-- NEW: Accepts maximum depth
):
    """
    [HELPER] Runs one greedy search path starting from a single CNOT.
    This version returns the best sequence found on this path, even if a later step is worse.
    """
    ng_circ1 = circ1.num_qubits
    best_cnot_sequence = [starting_cnot]
    
    # EFFICENCY FIX: Calculate the base circuit only ONCE
    temp_base_circuit = concatenate_circuits_with_separate_measurements(circ1, circ2)

    # Use the pre-calculated score passed in from the main function
    best_sequence_on_path = list(best_cnot_sequence)
    best_kl_on_path = initial_kl_for_path
    current_kl_sum_on_path = initial_kl_for_path
    improvement_made = True
    iteration_count = 1
    while improvement_made:
        iteration_count += 1
        improvement_made = False
        current_iteration_best_cnot = None
        # Logic is sound here: current_iteration_min_kl_sum starts as the path's current best
        current_iteration_min_kl_sum = current_kl_sum_on_path 
        
        for candidate_cnot in all_possible_single_cnots:
            if candidate_cnot not in best_cnot_sequence:
                temp_cnot_sequence = best_cnot_sequence + [candidate_cnot]
                
                # Use the pre-calculated base circuit
                temp_circuit_with_cnots = add_cnots_and_measurements_to_circuit(temp_base_circuit, ng_circ1, temp_cnot_sequence)
                
                kl_div1, kl_div2 = score_circuit_kl_divergences(temp_circuit_with_cnots, state_vec_probs_target1, state_vec_probs_target2, nshots)
                
                if kl_div1 is not None and kl_div2 is not None:
                    current_kl_sum = kl_div1 + kl_div2
                    
                    #if current_kl_sum < current_iteration_min_kl_sum:
                    if current_kl_sum / current_iteration_min_kl_sum < ratio_kl_tol:
                        current_iteration_min_kl_sum = current_kl_sum
                        current_iteration_best_cnot = candidate_cnot
                        improvement_made = True
                        
        if improvement_made:
            best_cnot_sequence.append(current_iteration_best_cnot)
            current_kl_sum_on_path = current_iteration_min_kl_sum
        
        # ROBUSTNESS FIX: Check for maximum depth limit first
        if len(best_cnot_sequence) >= max_cnot_depth:
             print(f"  Max depth ({max_cnot_depth}) reached. Stopping search path.")
             break

        # REFINED: Stagnation break condition (applies if no improvement was made)
        if not improvement_made:
            if len(best_cnot_sequence) > min_cnot_depth: 
                break

        if current_kl_sum_on_path < best_kl_on_path:
            best_kl_on_path = current_kl_sum_on_path
            best_sequence_on_path = list(best_cnot_sequence)
            print(f"    --> Found a better KL on this path: {best_kl_on_path:.6f} at depth {len(best_sequence_on_path)} with added CNOT {current_iteration_best_cnot}")

        if current_kl_sum_on_path < kl_tol:
            print(f"  KL Sum ({current_kl_sum_on_path:.6f}) below tolerance. Early stopping this epoch.")
            break
            
    return best_sequence_on_path, best_kl_on_path

def _run_greedy_removal_search(
    circ1, circ2, state_vec_probs_target1, state_vec_probs_target2,
    initial_cnot_sequence, min_kl_sum_initial, min_cnot_depth, nshots
):
    """
    [NEW HELPER] Runs a greedy search by removing CNOTs from an initial sequence.
    It iteratively removes the single CNOT whose absence most improves the KL divergence,
    stopping when removal no longer helps or the minimum depth is reached.
    """
    ng_circ1 = circ1.num_qubits
    
    # Use a copy of the initial sequence for manipulation
    best_cnot_sequence = list(initial_cnot_sequence) 
    
    # Initialize path tracking variables
    best_sequence_on_path = list(initial_cnot_sequence)
    best_kl_on_path = min_kl_sum_initial
    current_kl_sum_on_path = min_kl_sum_initial
    
    # Efficiency Fix: Calculate the base circuit only ONCE
    temp_base_circuit = concatenate_circuits_with_separate_measurements(circ1, circ2)

    print(f"  Starting removal search with {len(initial_cnot_sequence)} CNOTs (KL Sum: {best_kl_on_path:.6f})")

    # Loop continues as long as the sequence length is above the minimum required depth
    while len(best_cnot_sequence) > min_cnot_depth:
        improvement_made = False
        best_cnot_to_remove = None
        # Must use current_kl_sum_on_path as the threshold for improvement in this round
        best_kl_after_removal = current_kl_sum_on_path
        
        # --- Find the best CNOT to remove in this step (Greedy choice) ---
        # NOTE: If CNOTs are repeated, this loop finds the first one that improves the score.
        for i, cnot_to_remove in enumerate(best_cnot_sequence):
            # Create a temporary sequence by removing the CNOT at index i
            # This is safer than using list.remove(cnot) if identical CNOTs exist.
            temp_cnot_sequence = best_cnot_sequence[:i] + best_cnot_sequence[i+1:]
            
            temp_circuit_with_cnots = add_cnots_and_measurements_to_circuit(temp_base_circuit, ng_circ1, temp_cnot_sequence)
            kl_div1, kl_div2 = score_circuit_kl_divergences(temp_circuit_with_cnots, state_vec_probs_target1, state_vec_probs_target2, nshots)
            
            if kl_div1 is not None and kl_div2 is not None:
                current_kl_after_removal = kl_div1 + kl_div2
                
                # Check for absolute improvement over the current path score
                if current_kl_after_removal < best_kl_after_removal:
                    best_kl_after_removal = current_kl_after_removal
                    # Store the index for removal, which is unique
                    best_cnot_to_remove_index = i 
                    best_cnot_to_remove = cnot_to_remove # Store the CNOT identity for printing
                    improvement_made = True
        
        # --- Apply the best removal ---
        if improvement_made:
            # Safely remove the CNOT by index if the sequence contains duplicates
            best_cnot_sequence.pop(best_cnot_to_remove_index) 
            current_kl_sum_on_path = best_kl_after_removal
            
            # Update the overall best sequence found so far
            if current_kl_sum_on_path < best_kl_on_path:
                best_kl_on_path = current_kl_sum_on_path
                best_sequence_on_path = list(best_cnot_sequence)
            
            print(f"  Removed CNOT {best_cnot_to_remove}. New KL Sum: {current_kl_sum_on_path:.6f} at depth {len(best_cnot_sequence)}")

        else:
            print("  No CNOT removal improved the score. Stopping removal search.")
            break
            
    return best_sequence_on_path, best_kl_on_path


def find_best_cnot_sequence_brute_force_on_list(
    circ1: QuantumCircuit,
    circ2: QuantumCircuit,
    cnot_list_to_test: list,
    state_vec_probs_target1: dict,
    state_vec_probs_target2: dict,
    max_cnot_depth: int,
    nshots: int, 
    kl_tol: float = 0.01
):
    """
    [NEW HELPER] Performs a brute-force search on a given list of CNOT candidates.
    """
    ng_circ1 = circ1.num_qubits
    
    base_combined_circuit = concatenate_circuits_with_separate_measurements(circ1, circ2)
    base_circuit_with_measurements = add_cnots_and_measurements_to_circuit(base_combined_circuit, ng_circ1, [])
    kl_div1_no_cnot, kl_div2_no_cnot = score_circuit_kl_divergences(base_circuit_with_measurements, state_vec_probs_target1, state_vec_probs_target2, nshots)
    initial_kl_sum = kl_div1_no_cnot + kl_div2_no_cnot if kl_div1_no_cnot is not None and kl_div2_no_cnot is not None else float('inf')
    min_kl_sum = initial_kl_sum
    best_cnot_sequence = []
    
    print(f"\n--- Brute-force fallback: Testing permutations on {len(cnot_list_to_test)} candidates up to depth {max_cnot_depth} ---")
    start_time = time.time()
    for num_cnots in range(1, max_cnot_depth + 1):
        for cnot_combination in itertools.permutations(cnot_list_to_test, num_cnots):
            temp_cnot_sequence = list(cnot_combination)
            temp_base_circuit = concatenate_circuits_with_separate_measurements(circ1, circ2)
            temp_circuit_with_cnots = add_cnots_and_measurements_to_circuit(temp_base_circuit, ng_circ1, temp_cnot_sequence)
            kl_div1, kl_div2 = score_circuit_kl_divergences(temp_circuit_with_cnots, state_vec_probs_target1, state_vec_probs_target2, nshots)
            if kl_div1 is not None and kl_div2 is not None:
                current_kl_sum = kl_div1 + kl_div2
                if current_kl_sum < min_kl_sum:
                    min_kl_sum = current_kl_sum
                    best_cnot_sequence = temp_cnot_sequence
                    print(f"  --> New best sequence found: {best_cnot_sequence} with KL Sum: {min_kl_sum:.6f}")
                if min_kl_sum < kl_tol:
                    print(f"  KL sum ({min_kl_sum:.6f}) below tolerance {kl_tol}. Stopping early.")
                    return best_cnot_sequence, min_kl_sum
    
    end_time = time.time()
    print(f"Brute-Force CNOT search took: {end_time - start_time:.2f} seconds")
    
    if min_kl_sum >= initial_kl_sum: return [], initial_kl_sum
    else: return best_cnot_sequence, min_kl_sum

def find_best_cnot_sequence_multi_epoch(
    circ1: QuantumCircuit,
    circ2: QuantumCircuit,
    state_probs_initial1: dict,
    state_probs_initial2: dict,
    state_vec_probs_target1: dict,
    state_vec_probs_target2: dict,
    n_epochs: int = 10,
    min_cnot_depth: int = 1,
    nshots: int = 1000,
    threshold: float = 0.1,
    kl_tol: float = 0.01, 
    ratio_kl_tol: float = 0.6,
    max_greedy_depth: int = 30 # <-- NEW: Added max depth parameter
):
    """
    [NEW MAIN] Performs a multi-epoch greedy search with CNOT removal refinement.
    The brute-force fallback has been removed.
    """
    ng_circ1 = circ1.num_qubits
    ng_circ2 = circ2.num_qubits

    print(f"\n--- Starting Multi-Epoch Refined Search ---")
    start_total_time = time.time()
    
    # --- State and Density Matrix Analysis (Kept as is) ---
    state_list1_initial, _ = _process_target_state_input(state_probs_initial1)
    state_list2_initial, _ = _process_target_state_input(state_probs_initial2)
    state_list1_target, _ = _process_target_state_input(state_vec_probs_target1)
    state_list2_target, _ = _process_target_state_input(state_vec_probs_target2)

    combined_state_amplitudes0 = np.kron(state_list2_initial, state_list1_initial)
    sv0 = Statevector(combined_state_amplitudes0)
    combined_state_amplitudes = np.kron(state_list2_target, state_list1_target)
    sv = Statevector(combined_state_amplitudes)
    density_matrix_sv0 = DensityMatrix(sv0)
    density_matrix_sv = DensityMatrix(sv)
    subtracted_density_matrix = density_matrix_sv - density_matrix_sv0
    matrix_to_plot = subtracted_density_matrix.data
    
    # 1-4. Max, Min, Abs Average Print (Kept as is)
    max_element = np.max(matrix_to_plot.flatten())
    min_element = np.min(matrix_to_plot.flatten())
    abs_matrix = np.abs(matrix_to_plot.flatten())
    abs_average = np.mean(abs_matrix)
    print(f"Maximum Density Element: {max_element}")
    print(f"Minimum Density Element: {min_element}")
    print(f"Absolute Average: {abs_average}")

    # --- CNOT Candidate Pruning (Kept as is) ---
    row_indices_sv, col_indices_sv = np.where(matrix_to_plot > threshold)
    row_indices_sv0, col_indices_sv0 = np.where(matrix_to_plot < -threshold)
    dim = matrix_to_plot.shape[0]
    labels = [bin(i)[2:].zfill(int(np.log2(dim))) for i in range(dim)]
    row_labels_sv = [labels[i] for i in row_indices_sv]
    col_labels_sv = [labels[i] for i in col_indices_sv]
    all_labels_set = set(row_labels_sv) | set(col_labels_sv)
    sorted_labels = sorted(all_labels_set)
    row_labels_sv0 = [labels[i] for i in row_indices_sv0]
    col_labels_sv0 = [labels[i] for i in col_indices_sv0]
    all_labels_set0 = set(row_labels_sv0) | set(col_labels_sv0)
    sorted_labels0 = sorted(all_labels_set0)
    num_total_qubits_for_config = len(sorted_labels[0]) if sorted_labels else 0
    all_cnot_configurations = set()
    for ibit_string in sorted_labels0:
        for jbit_string in sorted_labels:
            locations_i_qiskit = [num_total_qubits_for_config - 1 - i for i, bit in enumerate(ibit_string) if bit == '1']
            locations_j_qiskit = [num_total_qubits_for_config - 1 - i for i, bit in enumerate(jbit_string) if bit == '1']
            new_cnot_pairs = create_cnot_pairs_from_locations(locations_i_qiskit, locations_j_qiskit)
            all_cnot_configurations.update(new_cnot_pairs)
    all_possible_single_cnots = sorted(list(all_cnot_configurations))
    
    print(f"Number of original CNOT candidates: {ng_circ1 * ng_circ2 * 2}")
    print(f"Number of refined CNOT candidates: {len(all_possible_single_cnots)}")

    # --- Initial KL Score (Kept as is) ---
    base_combined_circuit = concatenate_circuits_with_separate_measurements(circ1, circ2)
    base_circuit_with_measurements = add_cnots_and_measurements_to_circuit(base_combined_circuit, ng_circ1, [])
    kl_div1_no_cnot, kl_div2_no_cnot = score_circuit_kl_divergences(base_circuit_with_measurements, state_vec_probs_target1, state_vec_probs_target2, nshots)
    initial_kl_sum = kl_div1_no_cnot + kl_div2_no_cnot if kl_div1_no_cnot is not None and kl_div2_no_cnot is not None else float('inf')

    print(f"Initial KL divergence: {initial_kl_sum:.6f}")

    best_overall_sequence = []
    best_overall_kl_sum = initial_kl_sum
    
    if not all_possible_single_cnots:
        print("\nNo CNOT candidates found. Skipping all searches.")
    else:
        # --- Multi-Epoch greedy search (addition) ---
        num_epochs_to_run = min(n_epochs, len(all_possible_single_cnots))
        shuffled_candidates = all_possible_single_cnots.copy()
        random.shuffle(shuffled_candidates)
        
        for epoch in range(num_epochs_to_run):
            starting_cnot = shuffled_candidates[epoch]
            
            # EFFICENCY FIX: Calculate KL for the starting CNOT once
            temp_kl_scores = score_circuit_kl_divergences(
                add_cnots_and_measurements_to_circuit(base_combined_circuit, ng_circ1, [starting_cnot]), 
                state_vec_probs_target1, state_vec_probs_target2, nshots
            )

            if temp_kl_scores is not None: 
                temp_kl_sum = temp_kl_scores[0] + temp_kl_scores[1]
                print(f"\n--- Starting Epoch {epoch + 1}/{num_epochs_to_run} (Addition) with CNOT: {starting_cnot} (KL: {temp_kl_sum:.6f}) ---")
            else:
                print(f"\n--- Starting Epoch {epoch + 1}/{num_epochs_to_run} ---")
                continue

            # LOGIC FIX: Skip if the single starting CNOT doesn't beat the baseline by ratio
            if temp_kl_sum < initial_kl_sum:     
                print(f"  Skipping: Single CNOT ({temp_kl_sum:.6f}) failed to beat baseline ({initial_kl_sum:.6f}) by ratio {ratio_kl_tol}.")
                continue

            # CRUCIAL FIXES: Pass the pre-calculated KL and the max depth to the helper
            best_sequence_this_epoch, min_kl_this_epoch = _run_single_greedy_search_from_start(
                circ1, circ2, state_vec_probs_target1, state_vec_probs_target2,
                all_possible_single_cnots, starting_cnot, min_cnot_depth, nshots, kl_tol, ratio_kl_tol,
                initial_kl_for_path=temp_kl_sum, 
                max_cnot_depth=max_greedy_depth
            )
            
            print(f"  Epoch {epoch + 1} best KL Sum: {min_kl_this_epoch:.6f}")
            if min_kl_this_epoch < best_overall_kl_sum:
                best_overall_kl_sum = min_kl_this_epoch
                best_overall_sequence = best_sequence_this_epoch
                print(f"  --> Epoch {epoch + 1} found a new overall best KL Sum: {best_overall_kl_sum:.6f}")

        # --- Greedy CNOT removal search (Kept as is) ---
        if best_overall_sequence:
            print("\n--- Starting Greedy CNOT Removal Search ---")
            best_sequence_after_removal, best_kl_after_removal = _run_greedy_removal_search(
                circ1, circ2, state_vec_probs_target1, state_vec_probs_target2,
                best_overall_sequence, best_overall_kl_sum, 0, nshots
            )
            if best_kl_after_removal < best_overall_kl_sum:
                best_overall_kl_sum = best_kl_after_removal
                best_overall_sequence = best_sequence_after_removal
                print(f"\n--> Removal search found a new overall best KL Sum: {best_overall_kl_sum:.6f}")

    # --- Final Check (No Brute-Force Fallback) ---
    if not best_overall_sequence and best_overall_kl_sum >= initial_kl_sum:
        print(f"\nGreedy search failed to improve baseline KL sum ({best_overall_kl_sum:.6f}) or found no sequence.")
        print("Consider running an optional brute-force search.")
    
    end_total_time = time.time()
    print(f"\nTotal search took: {end_total_time - start_total_time:.2f} seconds.")
    
    return best_overall_sequence, best_overall_kl_sum

# --- Function to normalize dictionary values ---
def vector_normalize_dictionary_values(input_dict):
    """Normalizes the values of a dictionary by dividing each value by the L2 norm of all values."""
    values = np.array(list(input_dict.values()), dtype=float)
    if values.size == 0:  # Handle empty dictionary
        return {}
    
    norm_val = np.linalg.norm(values)  # Calculate the L2 norm of the values

    if norm_val == 0:  # Avoid division by zero if all values are zero
        return {key: 0.0 for key in input_dict}

    normalized_dict = {}
    # Iterate through original dictionary items to maintain key order
    for i, (key, value) in enumerate(input_dict.items()):
        normalized_dict[key] = values[i] / norm_val
    return normalized_dict

def _single_cnot_insertion_search(
    base_circuit: QuantumCircuit,
    ng_circ1: int,
    current_sequence: list,
    candidate_cnots: list,
    target_probs1: dict,
    target_probs2: dict,
    nshots: int,
    kl_tol: float, 
    ratio_kl_tol: float
):
    """
    Finds the best single CNOT to insert into the current sequence at the best location.
    Returns the new sequence, its KL sum, and the added CNOT.
    """
    #print("   - Entering single CNOT insertion search -")
    best_single_cnot = None
    best_insertion_index = -1
    min_kl_after_add = float('inf')
    n_trials = 0
    
    remaining_cnots = [c for c in candidate_cnots if c not in current_sequence]
    if not remaining_cnots:
        print("   - No remaining CNOTs to add. Ending single-CNOT search.")
        return current_sequence, float('inf'), None

    for cnot_to_add in remaining_cnots:
        # Try inserting the CNOT at every possible position in the current sequence
        for i in range(len(current_sequence) + 1):
            n_trials += 1
            trial_sequence = current_sequence[:i] + [cnot_to_add] + current_sequence[i:]
            
            trial_circuit = add_cnots_and_measurements_to_circuit(base_circuit, ng_circ1, trial_sequence)
            kl_divs = score_circuit_kl_divergences(trial_circuit, target_probs1, target_probs2, nshots)
            
            if kl_divs is not None:
                kl_sum = kl_divs[0] + kl_divs[1]
                
                if kl_sum / min_kl_after_add < ratio_kl_tol:                   
                    min_kl_after_add = kl_sum
                    best_single_cnot = cnot_to_add
                    best_insertion_index = i
                    
    
    if best_single_cnot is not None:
        new_sequence = current_sequence[:best_insertion_index] + [best_single_cnot] + current_sequence[best_insertion_index:]
        print(f"  - No. CNOT tested insertion: {n_trials} | Added CNOT {best_single_cnot} at index {best_insertion_index} | Best added KL sum: {min_kl_after_add:.6f}")
        return new_sequence, min_kl_after_add, best_single_cnot
    else:
        print(f"  - No. CNOT tested insertion: {n_trials}")

    return current_sequence, float('inf'), None

def _single_cnot_deletion_search(
    base_circuit: 'QuantumCircuit',
    ng_circ1: int,
    current_sequence: list,
    candidate_cnots: list, # Argument included for signature consistency, but unused in deletion logic
    target_probs1: dict,
    target_probs2: dict,
    nshots: int,
    kl_tol: float, # Argument included for signature consistency, but unused in deletion logic
    ratio_kl_tol: float # Argument included for signature consistency, but unused in deletion logic
):
    """
    Finds the single CNOT whose removal yields the lowest KL divergence sum.
    Returns the new sequence, its KL sum, and the deleted CNOT.
    """
    if not current_sequence:
        print("   - Cannot perform deletion on an empty sequence.")
        return current_sequence, float('inf'), None

    # Calculate baseline KL sum of the current sequence for comparison
    base_circuit_kl = add_cnots_and_measurements_to_circuit(base_circuit, ng_circ1, current_sequence)
    kl_divs_base = score_circuit_kl_divergences(base_circuit_kl, target_probs1, target_probs2, nshots)
    base_kl_sum = kl_divs_base[0] + kl_divs_base[1] if kl_divs_base is not None else float('inf')
    
    # Initialize candidates for deletion
    # Note: We initialize min_kl_after_del to the current best KL sum. 
    # This means the best deletion must result in a KL sum strictly better than the current one.
    min_kl_after_del = base_kl_sum 
    best_deletion_index = -1
    deleted_cnot = None
    n_trials = 0

    # Try deleting every CNOT in the current sequence
    for i in range(len(current_sequence)):
        n_trials += 1
        
        # Create the trial sequence by excluding the CNOT at index i
        trial_sequence = current_sequence[:i] + current_sequence[i+1:]
        
        trial_circuit = add_cnots_and_measurements_to_circuit(base_circuit, ng_circ1, trial_sequence)
        kl_divs = score_circuit_kl_divergences(trial_circuit, target_probs1, target_probs2, nshots)
        
        if kl_divs is not None:
            kl_sum = kl_divs[0] + kl_divs[1]
            
            # Look for the lowest KL sum
            if kl_sum / min_kl_after_del < ratio_kl_tol:
                min_kl_after_del = kl_sum
                best_deletion_index = i
                
    if best_deletion_index != -1:
        deleted_cnot = current_sequence[best_deletion_index]
        new_sequence = current_sequence[:best_deletion_index] + current_sequence[best_deletion_index+1:]
        print(f"  - No. CNOT tested deletion: {n_trials} | Deleted CNOT {deleted_cnot} at index {best_deletion_index} | Best deletion KL sum: {min_kl_after_del:.6f}")
        return new_sequence, min_kl_after_del, deleted_cnot
    else:
        print(f"  - No. CNOT tested deletion: {n_trials}. Deletion did not improve KL sum.")
        # Return the original sequence and a signal that no CNOT was removed
        return current_sequence, base_kl_sum, None
    

def _pairwise_addition_search(
    base_circuit: 'QuantumCircuit',
    ng_circ1: int,
    initial_sequence: list,
    candidate_cnots: list,
    target_probs1: dict,
    target_probs2: dict,
    nshots: int,
    kl_tol: float = 0.01,
    nchoose: int = 2,
    ratio_kl_tol: float = 0.6
    ):
    """
    Iteratively adds the best CNOT pair or single CNOT based on which provides
    the greater reduction in KL divergence.
    Returns the final sequence and its corresponding KL sum.
    """
    print("  - Starting CNOT search -")
    current_sequence = list(initial_sequence)
    
    # Calculate baseline KL with the initial sequence
    base_circuit_kl = add_cnots_and_measurements_to_circuit(base_circuit, ng_circ1, current_sequence)
    kl_divs_base = score_circuit_kl_divergences(base_circuit_kl, target_probs1, target_probs2, nshots)
    best_kl_sum = kl_divs_base[0] + kl_divs_base[1] if kl_divs_base is not None else float('inf')
    
    improvement_made = True
    min_kl_of_candidates = best_kl_sum
    while improvement_made:
        improvement_made = False
        
        # Initialize best candidates for this iteration
        best_candidate_sequence = None
        
        remaining_cnots = [c for c in candidate_cnots if c not in current_sequence]

        # --- PHASE 1: Find the best PAIR to add ---
        if len(remaining_cnots) >= nchoose:
            best_pair_to_add = None
            min_kl_after_pair_add = float('inf')
            n_trials_pair = 0

            for pair in itertools.permutations(remaining_cnots, nchoose):
                n_trials_pair += 1
                trial_sequence = current_sequence + list(pair)
                trial_circuit = add_cnots_and_measurements_to_circuit(base_circuit, ng_circ1, trial_sequence)
                kl_divs = score_circuit_kl_divergences(trial_circuit, target_probs1, target_probs2, nshots)
                
                if kl_divs is not None:
                    kl_sum = kl_divs[0] + kl_divs[1]

                    #if kl_sum < min_kl_of_candidates:
                    if kl_sum / min_kl_of_candidates < ratio_kl_tol:
                        min_kl_after_pair_add = kl_sum
                        best_pair_to_add = list(pair)
            
            if best_pair_to_add is not None:
                min_kl_of_candidates = min_kl_after_pair_add
                best_candidate_sequence = current_sequence + best_pair_to_add
                print(f"  - No. tested pairs: {n_trials_pair} | Added CNOTs {best_pair_to_add} | Best pair KL sum: {min_kl_of_candidates:.6f}")
            else: 
                print(f"  - No. tested pairs: {n_trials_pair} | No added CNOTs...")

        # --- PHASE 2: Find the best SINGLE CNOT to add ---
        new_sequence_single, new_kl_sum_single, added_cnot = _single_cnot_insertion_search(
            base_circuit, ng_circ1, current_sequence, candidate_cnots,
            target_probs1, target_probs2, nshots, kl_tol, ratio_kl_tol
        )

        if added_cnot is not None:
            if new_kl_sum_single < min_kl_of_candidates:
                min_kl_of_candidates = new_kl_sum_single
                best_candidate_sequence = new_sequence_single
                print(f"  - Best single CNOT insertion KL Sum: {min_kl_of_candidates:.6f}")
            else:
                print("  - Single CNOT insertion did not improve upon the best pair.")

        # --- PHASE 3: Find the best SINGLE CNOT to add ---
        new_sequence_del, new_kl_sum_del, deleted_cnot = _single_cnot_deletion_search(
        base_circuit, ng_circ1, current_sequence, candidate_cnots,
        target_probs1, target_probs2, nshots, kl_tol, ratio_kl_tol
        )
        
        if deleted_cnot is not None:
            if new_kl_sum_del < min_kl_of_candidates:
                min_kl_of_candidates = new_kl_sum_del
                best_candidate_sequence = new_sequence_del
                print(f"  - Best single CNOT pruning KL Sum: {min_kl_of_candidates:.6f}")
            else:
                print("  - Single CNOT pruning did not improve upon the best pair.")


        # --- PHASE 3: Update the sequence if a better candidate was found ---
        if best_candidate_sequence is not None and min_kl_of_candidates / best_kl_sum < ratio_kl_tol:
            current_sequence = best_candidate_sequence
            best_kl_sum = min_kl_of_candidates
            improvement_made = True
            print(f"  - Adopted new sequence. New KL Sum: {best_kl_sum:.6f}")
        else:
            print("  - No significant improvement found from pairs or single CNOTs. Ending search.")
            break
        
        if best_kl_sum < kl_tol:
            print(f"  - CNOT configuration met KL tolerance...")
            break
                
    return current_sequence, best_kl_sum

def find_best_cnot_sequence_iterative_n_wise(
    circ1: QuantumCircuit,
    circ2: QuantumCircuit,
    state_probs_initial1: dict,
    state_probs_initial2: dict,
    state_vec_probs_target1: dict,
    state_vec_probs_target2: dict,
    nshots: int = 1000,
    threshold: float = 0.1,
    nchoose: int = 2,
    kl_tol: float = 0.01, 
    ratio_kl_tol: float = 0.6
    ):
    """
    Performs an iterative optimization loop by separating pairwise addition
    and removal phases.
    """
    import time
    
    ng_circ1 = circ1.num_qubits

    print(f"\n--- Starting Iterative Pairwise Search ---")
    start_total_time = time.time()

    # --- Step 1: Build refined CNOT candidate list ---
    state_list1_initial, _ = _process_target_state_input(state_probs_initial1)
    state_list2_initial, _ = _process_target_state_input(state_probs_initial2)
    state_list1_target, _ = _process_target_state_input(state_vec_probs_target1)
    state_list2_target, _ = _process_target_state_input(state_vec_probs_target2)

    combined_state_amplitudes0 = np.kron(state_list2_initial, state_list1_initial)
    combined_state_amplitudes = np.kron(state_list2_target, state_list1_target)
    sv0 = Statevector(combined_state_amplitudes0)
    sv = Statevector(combined_state_amplitudes)
    dm0 = DensityMatrix(sv0)
    dm = DensityMatrix(sv)
    diff = dm - dm0
    matrix_data = diff.data

    # 1-4. Max, Min, Abs Average Print (Kept as is)
    max_element = np.max(matrix_data.flatten())
    min_element = np.min(matrix_data.flatten())
    abs_matrix = np.abs(matrix_data.flatten())
    abs_average = np.mean(abs_matrix)
    print(f"Maximum Density Element: {max_element}")
    print(f"Minimum Density Element: {min_element}")
    print(f"Absolute Average: {abs_average}")

    row_indices, col_indices = np.where(matrix_data > threshold)
    row_indices0, col_indices0 = np.where(matrix_data < -threshold)

    dim = matrix_data.shape[0]
    labels = [bin(i)[2:].zfill(int(np.log2(dim))) for i in range(dim)]
    active_labels = set(labels[i] for i in row_indices) | set(labels[i] for i in col_indices)
    active_labels0 = set(labels[i] for i in row_indices0) | set(labels[i] for i in col_indices0)

    all_cnot_configurations = set()
    for ibit_string in active_labels0:
        for jbit_string in active_labels:
            i_locs = [len(ibit_string) - 1 - i for i, bit in enumerate(ibit_string) if bit == '1']
            j_locs = [len(jbit_string) - 1 - i for i, bit in enumerate(jbit_string) if bit == '1']
            all_cnot_configurations.update(create_cnot_pairs_from_locations(i_locs, j_locs))
    initial_cnot_config = sorted(list(all_cnot_configurations))

    print(f"Number of initial CNOT candidates: {len(initial_cnot_config)}")
    
    # Optional: Plotting Real and Imaginary parts separately (also without annotations)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    sns.heatmap(np.real(matrix_data),
                annot=False,
                cmap="RdBu",
                cbar=True,
                xticklabels=labels,
                yticklabels=labels,
                ax=axes[0])
    axes[0].set_title("Real Part of Subtracted Density Matrix")
    axes[0].set_xlabel("Column Index (Binary State)")
    axes[0].set_ylabel("Row Index (Binary State)")

    sns.heatmap(np.imag(matrix_data),
                annot=False,
                cmap="coolwarm",
                cbar=True,
                xticklabels=labels,
                yticklabels=labels,
                ax=axes[1])
    axes[1].set_title("Imaginary Part of Subtracted Density Matrix")
    axes[1].set_xlabel("Column Index (Binary State)")
    axes[1].set_ylabel("Row Index (Binary State)")

    plt.tight_layout()
    plt.savefig('dens_state_diff.svg')
    plt.close(fig)

    # --- Step 2: Baseline (no CNOTs) ---
    base_combined_circuit = concatenate_circuits_with_separate_measurements(circ1, circ2)
    base_circuit = add_cnots_and_measurements_to_circuit(base_combined_circuit, ng_circ1, [])
    kl_divs = score_circuit_kl_divergences(base_circuit, state_vec_probs_target1, state_vec_probs_target2, nshots)
    initial_kl_sum = kl_divs[0] + kl_divs[1] if kl_divs is not None else float('inf')

    print(f"Initial KL-divergence: {initial_kl_sum:.6f}")

    if len(initial_cnot_config) < 2:
        print("Not enough CNOT candidates to form a pair. Skipping search.")
        return initial_cnot_config, [], [], initial_kl_sum

    # --- Step 3: Pairwise Addition Phase ---
    best_add_sequence, best_add_kl_sum = _pairwise_addition_search(
        base_combined_circuit, ng_circ1, [], initial_cnot_config,
        state_vec_probs_target1, state_vec_probs_target2, nshots, kl_tol, nchoose, ratio_kl_tol
    )
    print(f"\n--- Pairwise Addition Result ---")
    print(f"Best sequence after addition: {best_add_sequence}")
    print(f"KL sum after addition: {best_add_kl_sum:.6f}")

    end_time = time.time()
    print(f"\nTotal search time: {end_time - start_total_time:.2f} seconds.")

    # Return the configurations as requested
    return (
        initial_cnot_config,
        best_add_sequence,
        best_add_kl_sum
    )