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


def plot_measurement_histograms(circuit: QuantumCircuit, nshots: int = 1000, title_prefix: str = "", figure_save_name: str = None):
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
            fig, axes = plt.subplots(1, 2, figsize=(12, 5)) # 1 row, 2 columns
            fig.suptitle(f"{title_prefix} - Measurement Counts ({nshots} shots)", fontsize=16)

            if counts_measure1 is not None:
                plot_histogram(counts_measure1, ax=axes[0], title="c_measure1")
            else:
                axes[0].set_title("c_measure1 (Not Found)")
                axes[0].text(0.5, 0.5, "No data", horizontalalignment='center', verticalalignment='center', transform=axes[0].transAxes)

            if counts_measure2 is not None:
                plot_histogram(counts_measure2, ax=axes[1], title="c_measure2")
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


    