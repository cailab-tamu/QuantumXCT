import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import random
import time
from collections import defaultdict
from typing import List, Union, Set, Dict, Tuple, Optional, Callable
import pandas as pd
import networkx as nx
from matplotlib.patches import Ellipse

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector, DensityMatrix
from qiskit.circuit.library import Initialize
from qiskit.providers import Backend
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime import EstimatorV2 as Estimator

from scipy.special import rel_entr
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Global seed for reproducibility (default 42).
# Seeds Python's `random`, NumPy, and the default AerSimulator (seed_simulator),
# so topology search (candidate shuffle) and shot sampling are reproducible.
# ---------------------------------------------------------------------------
_GLOBAL_SEED = 42


def set_global_seed(seed: int = 42) -> int:
    """Set the global seed used across the package (random, numpy, Aer sampling)."""
    global _GLOBAL_SEED
    _GLOBAL_SEED = int(seed)
    random.seed(_GLOBAL_SEED)
    np.random.seed(_GLOBAL_SEED)
    return _GLOBAL_SEED


def _default_backend(seed: Optional[int] = None):
    """A local AerSimulator seeded for reproducible sampling."""
    return AerSimulator(seed_simulator=_GLOBAL_SEED if seed is None else int(seed))


# --- Modified create_initial_circuit function ---
def build_initial_state_circuit(sparse_amplitude_dict, norm: Optional[str] = None):
    """
    Creates a quantum circuit initialized to a given state vector.
    This function accepts a sparse dictionary, infers num_qubits,
    pads non-specified states with zeros, and normalizes the state vector internally.

    Args:
        sparse_amplitude_dict (dict): A dictionary where keys are bitstrings
                                      (e.g., '000', '101') and values are either
                                      pre-computed amplitudes (``norm=None``) or raw
                                      counts / frequencies (``norm='l1'`` or ``'l2'``).
        norm (str | None): Encoding convention used to turn the values into
                           amplitudes. This is the single L1/L2 switch.
                             - None (default): values are treated as amplitudes and
                               only L2-safety-normalized (idempotent for a valid state).
                               Use this when the dict was already produced by
                               ``vector_normalize_dictionary_values``.
                             - 'l1': values are raw counts; amplitude = sqrt(count / sum(count))
                               so measurement prob is proportional to the frequency
                               (frequency-preserving encoding — QuantumXCT default).
                             - 'l2': values are raw counts; amplitude = count / ||count||_2
                               so measurement prob is proportional to count^2
                               (original paper encoding, emphasizes common states).
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

    # 4. Normalize the full state vector into amplitudes according to `norm`.
    if norm == "l1":
        # Values are raw counts; encode so that |amplitude|^2 == frequency.
        total = full_state_vector.sum()
        if total <= 0:
            state_vec_probs = full_state_vector
        else:
            state_vec_probs = np.sqrt(full_state_vector / total)
    else:
        # norm is None ('values are already amplitudes') or 'l2' ('raw counts,
        # encode as count / ||count||_2'). Both use the L2 path, which is a
        # no-op (idempotent) for an already-normalized amplitude vector.
        if norm not in (None, "l2"):
            raise ValueError(f"Unknown norm '{norm}', use None, 'l1' or 'l2'.")
        norm_val = np.linalg.norm(full_state_vector)
        if norm_val == 0:
            # If all values are zero, the normalized vector is also all zeros.
            # This represents the |0...0> state.
            state_vec_probs = full_state_vector
        else:
            state_vec_probs = full_state_vector / norm_val

    # Verify L2-normalization for good measure (a valid state must have unit norm).
    total_norm = np.linalg.norm(state_vec_probs)
    if not np.isclose(total_norm, 1.0) and total_norm != 0:
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

def apply_entanglement_topology(
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

        circuit_with_cnots.crx(np.pi/2, qr_all[control_q], qr_all[target_q]) # Look for mixed potential states 
        #circuit_with_cnots.cx(qr_all[control_q], qr_all[target_q]) # Look for entangled 
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
    #circuit_with_cry.measure(qr_all[0:circ1_num_qubits], cr_measure1)
    #circuit_with_cry.measure(qr_all[circ1_num_qubits:circuit_with_crx.num_qubits], cr_measure2)

    return circuit_with_crx


# --- Function to normalize dictionary values ---
def vector_normalize_dictionary_values(data_dict: dict, norm: str = "l1") -> dict:
    """
    Convert a count/frequency dictionary into state-vector amplitudes.
    norm = "l2"  : amplitude = count / ||count||_2   (measurement prob ∝ count^2,
                   original paper encoding, Eqs. 1–2)   [emphasizes common states]
    norm = "l1"  : amplitude = sqrt( count / sum(count) )  (measurement prob ∝ count,
                   i.e., square-root / frequency-preserving encoding)
    """
    if not data_dict:
        return {}

    values = np.array(list(data_dict.values()), dtype=float)

    if norm == "l2":
        denom = np.linalg.norm(values)              # existing behavior
        amps = values / denom if denom != 0 else values
    elif norm == "l1":
        total = values.sum()
        probs = values / total if total != 0 else values
        amps = np.sqrt(probs)                        # sqrt so that |amp|^2 = frequency
    else:
        raise ValueError(f"Unknown norm '{norm}', use 'l2' or 'l1'.")

    return {k: a for k, a in zip(data_dict.keys(), amps)}


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
        p_val = max(p_dist.get(key, 0.0), epsilon) # Ensure P is never 0
        q_val = max(q_dist.get(key, 0.0), epsilon) # Ensure Q is never 0
        kl_div += p_val * np.log(p_val / q_val)
                
    return kl_div

# --- NEW HELPER FUNCTION ---
def _normalize_amplitudes(full_vec: np.ndarray, norm: Optional[str]) -> np.ndarray:
    """
    Turns a raw (or already-amplitude) vector into a unit-norm amplitude vector,
    using the same L1/L2 convention as ``build_initial_state_circuit``.

      - None / 'l2' : amplitude = value / ||value||_2  (idempotent for a valid state)
      - 'l1'        : amplitude = sqrt(value / sum(value))  (frequency-preserving)
    """
    if norm == "l1":
        real_vals = full_vec.real
        total = real_vals.sum()
        if total <= 0:
            return full_vec
        return np.sqrt(np.clip(real_vals / total, 0.0, None)).astype(complex)
    if norm not in (None, "l2"):
        raise ValueError(f"Unknown norm '{norm}', use None, 'l1' or 'l2'.")
    n = np.linalg.norm(full_vec)
    return full_vec / n if n != 0 else full_vec


def _process_target_state_input(
    target_input: Union[dict, list, np.ndarray],
    norm: Optional[str] = None,
) -> Tuple[np.ndarray, int]:
    """
    Converts a dictionary, list, or array to a normalized complex NumPy array.

    Args:
        target_input: bitstring->value dict, or a dense list/array.
        norm: L1/L2 encoding convention (see ``build_initial_state_circuit``).
              None means the input is already amplitudes (L2-safety only); 'l1'/'l2'
              treat the input as raw counts and must match the convention used to
              build the initial-state circuits, otherwise the KL comparison is
              between two different encodings.
    """
    if isinstance(target_input, dict):
        if not target_input: return np.array([], dtype=complex), 0
        sample_bitstring = next(iter(target_input))
        n_qubits = len(sample_bitstring)
        vector_len = 2**n_qubits
        full_vec = np.zeros(vector_len, dtype=complex) # Use complex dtype
        for bs, val in target_input.items():
            idx = int(bs, 2)
            full_vec[idx] = complex(val) # Cast to complex
        return _normalize_amplitudes(full_vec, norm), n_qubits
    elif isinstance(target_input, (list, np.ndarray)):
        if len(target_input) == 0: return np.array([], dtype=complex), 0
        processed_vec = np.array(target_input, dtype=complex) # Use complex dtype
        n_qubits = int(np.log2(len(processed_vec)))
        return _normalize_amplitudes(processed_vec, norm), n_qubits
    else:
        raise TypeError("Target state must be a dictionary, list, or numpy array.")


def score_circuit_kl_divergences(
    circuit_to_evaluate: QuantumCircuit,
    state_vec_probs_target1: list or np.ndarray,
    state_vec_probs_target2: list or np.ndarray,
    nshots: int = 1000,
    backend=None,
    error_mitigation: bool = False,
    norm: Optional[str] = None
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
        backend: A Qiskit backend. If None, defaults to a local AerSimulator.
                 Pass a real IBM backend here to run on hardware.
        error_mitigation (bool): If True, enables dynamical decoupling and twirling
                                 on the sampler (intended for real-hardware runs).

    Returns:
        tuple: A tuple containing (kl_div1, kl_div2), which are the KL divergence
               values for c_measure1 and c_measure2 respectively.
               Returns (None, None) if an error occurs during simulation or result retrieval.
    """
    # 1. Define target states (convert amplitudes to probability distributions)   
    # Process the input target states to ensure they are full, normalized NumPy arrays
    processed_state_vec_probs_target1, num_qubits_target1 = _process_target_state_input(state_vec_probs_target1, norm)
    processed_state_vec_probs_target2, num_qubits_target2 = _process_target_state_input(state_vec_probs_target2, norm)

    # 2. Define target states (convert amplitudes to probability distributions)
    # These lines now correctly operate on the processed (dense, numerically indexed) arrays
    prob_dist_target1 = {bin(i)[2:].zfill(num_qubits_target1): np.abs(processed_state_vec_probs_target1[i])**2
                         for i in range(2**num_qubits_target1)}

    prob_dist_target2 = {bin(i)[2:].zfill(num_qubits_target2): np.abs(processed_state_vec_probs_target2[i])**2
                         for i in range(2**num_qubits_target2)}

    # 5. Simulate / execute the circuit
    if backend is None:
        backend = _default_backend()

    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    qc_comp = pm.run(circuit_to_evaluate.decompose())

    sampler = Sampler(mode=backend)
    if error_mitigation:
        # Only meaningful on real hardware; harmless flags for a demonstration run.
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.twirling.enable_gates = True
        sampler.options.twirling.enable_measure = True

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


def plot_measurement_histograms(
    circuit: QuantumCircuit,
    nshots: int = 1000,
    backend=None,
    title_prefix: str = "",
    figure_save_name: str = None,
    figsize: Tuple[int, int] = (12, 5)
):
    """
    Runs the given circuit on a specified Qiskit backend (simulator or hardware)
    and plots measurement histograms for its classical registers 'c_measure1'
    and 'c_measure2' side-by-side.

    Args:
        circuit (QuantumCircuit): The circuit to execute and plot. Should contain classical registers
                                  named 'c_measure1' and 'c_measure2'.
        nshots (int, optional): Number of shots (circuit runs). Defaults to 1000.
        backend (optional): Qiskit backend to run the circuit (e.g. AerSimulator, or IBM/Q device).
                            If None, uses AerSimulator by default.
        title_prefix (str, optional): Prefix for the figure title.
        figure_save_name (str, optional): If provided, saves the figure to this filename.
        figsize (tuple, optional): Figure size in inches. Default is (12, 5).

    Returns:
        Tuple (counts_measure1, counts_measure2): Measured bitstring counts for both registers.

    Notes:
        - If 'c_measure1' or 'c_measure2' registers are missing, their count/histogram will be skipped.
        - To run on hardware, pass a backend instance provisioned through Qiskit.
    """

    print(f"\n--- Running circuit for: {title_prefix} ---")
    # 1. Select backend
    if backend is None:
        backend = _default_backend()
    # transpile if using AerSimulator (for real hardware, sometimes needed as well)
    try:
        pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
        qc_comp = pm.run(circuit)
    except Exception:
        # Fallback if not available for the backend (some HW backends)
        qc_comp = circuit
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
                    fig.savefig(figure_save_name, dpi=300)
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


def optimize_crx_angles(
    circ1: QuantumCircuit,
    circ2: QuantumCircuit,
    state_vec_probs_target1: list or np.ndarray,
    state_vec_probs_target2: list or np.ndarray,
    cnot_topology: list[tuple[int, int]], # The fixed CNOT structure (control, target)
    nshots: int = 1000,
    etol: float = 1e-6,
    opt_method: str = 'COBYLA', # Optimization method to use "L-BFGS-B", "COBYLA", etc.
    initial_angle_value: float = np.pi / 2 # Start angle optimization at the topology-search value CRX(pi/2), then fine-tune (starting at 0 = identity gates tends to get stuck near baseline)
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
        initial_angle_value (float): The initial angle value for all CRX gates. Defaults to pi/2,
                              which starts the optimizer exactly at the topology-search
                              configuration (CRX(pi/2)) and then fine-tunes. Starting at 0
                              makes every CRX the identity and tends to get stuck near baseline.

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
        base_circuit_with_measurements = apply_entanglement_topology(
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

    base_combined_circuit = concatenate_circuits_with_separate_measurements(circ1, circ2)
    # Define the objective function for the optimizer
    def objective_function(angles_array):
        # Ensure angles_array is a list for add_crx_gates_and_measurements_to_circuit
        angles_list = angles_array.tolist()

        # Build the circuit with CRX gates and current angles
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
    #bounds = [(-np.pi, np.pi)] * num_crx_gates
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


def compare_angle_initializations(
    circ1: QuantumCircuit,
    circ2: QuantumCircuit,
    state_vec_probs_target1,
    state_vec_probs_target2,
    cnot_topology: list,
    inits=(0.0, np.pi / 2),
    nshots: int = 5000,
    opt_method: str = "COBYLA",
    etol: float = 1e-6,
    norm: Optional[str] = None,
    show_plot: bool = True,
    filename: Optional[str] = None,
):
    """Run CRX-angle optimization from several starting angles and plot convergence.

    Optimizes the same fixed topology once per value in ``inits`` (e.g. 0 and pi/2),
    capturing each run's per-evaluation KL history, and overlays the convergence
    curves on one figure. Before each run the global seed is reset, so all
    initializations see the SAME shot-noise realization (a fair comparison).

    Returns:
        dict keyed by init value -> {"angles", "final_kl", "history"}.
    """
    results = {}
    for init in inits:
        set_global_seed(_GLOBAL_SEED)   # common random numbers across initializations
        angles, kl, hist = optimize_crx_angles(
            circ1, circ2, state_vec_probs_target1, state_vec_probs_target2,
            cnot_topology, nshots=nshots, etol=etol, opt_method=opt_method,
            initial_angle_value=init,
        )
        results[init] = {"angles": angles, "final_kl": kl, "history": hist}

    if show_plot or filename:
        import matplotlib.pyplot as plt   # lazy import (avoids GUI-DLL load at import time)
        plt.figure(figsize=(7, 5))
        for init, r in results.items():
            hist = r["history"]
            if not hist:
                continue
            plt.plot(range(1, len(hist) + 1), hist, marker="o", markersize=3,
                     linewidth=1.6, label=f"init={init:.4f}  (final KL {r['final_kl']:.4f})")
        plt.xlabel("objective evaluation")
        plt.ylabel("combined KL divergence")
        plt.title("CRX angle-optimization convergence by initialization")
        plt.grid(True, linestyle=":", alpha=0.5)
        plt.legend(frameon=True)
        plt.tight_layout()
        if filename:
            plt.savefig(filename, bbox_inches="tight", dpi=300)
        if show_plot:
            plt.show()
        plt.close()

    return results


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

def find_cnot_candidates_from_state_diff(
    state_probs_initial1: Union[dict, list, np.ndarray],
    state_probs_initial2: Union[dict, list, np.ndarray],
    state_vec_probs_target1: Union[dict, list, np.ndarray],
    state_vec_probs_target2: Union[dict, list, np.ndarray],
    threshold: float,
    search_mode: str = "pos",  # Now defaults to "pos" based on our discussion
    plot_filename: Optional[str] = None,
    show_plot: bool = False,
    verbose_print: bool = False,
    norm: Optional[str] = None
) -> Tuple[List[Tuple[int, int]], Dict[str, float]]:

    # --- 1. Process Inputs (same L1/L2 convention as the circuits being scored) ---
    state_list1_initial, _ = _process_target_state_input(state_probs_initial1, norm)
    state_list2_initial, _ = _process_target_state_input(state_probs_initial2, norm)
    state_list1_target, _ = _process_target_state_input(state_vec_probs_target1, norm)
    state_list2_target, _ = _process_target_state_input(state_vec_probs_target2, norm)

    # Initial = Basal, Target = Interacting
    combined_amps0 = np.kron(state_list2_initial, state_list1_initial)
    combined_amps = np.kron(state_list2_target, state_list1_target)
    
    if combined_amps0.size == 0 or combined_amps.size == 0:
        return [], {}

    # --- 2. Matrix Math ---
    # Large systems (> 12 qubits): the dense 2^N x 2^N difference matrix needs
    # 16 * 4^N bytes (17 TB at N = 20).  Use the exact sparse equivalent instead
    # (same candidates; see find_cnot_candidates_sparse).  Small systems and any call
    # that asks for the heatmap keep the original dense path, so all saved
    # results are unchanged.
    if combined_amps0.size > 2 ** 12 and not (plot_filename or show_plot):
        cnot_candidates, stats = find_cnot_candidates_sparse(
            state_probs_initial1, state_probs_initial2, state_vec_probs_target1, state_vec_probs_target2,
            threshold=threshold, search_mode=search_mode, norm=norm, precision="float32")
        if verbose_print:
            nq = int(np.log2(combined_amps.size))
            print(f"\n--- CNOT Analysis ({search_mode}, sparse) ---")
            print(f"Max Signal: {stats['max_element']:.4f}")
            print(f"Candidates Found: {len(cnot_candidates)} / {nq * (nq - 1)}")
        return cnot_candidates, stats

    dm0 = DensityMatrix(Statevector(combined_amps0))
    dm = DensityMatrix(Statevector(combined_amps))
    diff_matrix = (dm - dm0).data
    real_part_data = diff_matrix.real
    
    dim = diff_matrix.shape[0]
    num_qubits = int(np.log2(dim))
    labels = [bin(i)[2:].zfill(num_qubits) for i in range(dim)]

    # --- 3. Mode-Aware Statistics ---
    pos_mask = real_part_data > 0
    neg_mask = real_part_data < 0
    
    stats = {
        "max_element": np.max(real_part_data),
        "min_element": np.min(real_part_data),
        "abs_average": np.mean(np.abs(diff_matrix)),
    }

    if search_mode == "pos":
        stats["relevant_avg"] = np.mean(real_part_data[pos_mask]) if np.any(pos_mask) else 0.0
    elif search_mode == "neg":
        stats["relevant_avg"] = np.mean(real_part_data[neg_mask]) if np.any(neg_mask) else 0.0
    else: # abs
        stats["avg_positive"] = np.mean(real_part_data[pos_mask]) if np.any(pos_mask) else 0.0
        stats["avg_negative"] = np.mean(real_part_data[neg_mask]) if np.any(neg_mask) else 0.0

    # --- 4. CNOT Candidate Logic (Fixed for Diagonal Peaks) ---
    all_cnot_configurations = set()
    
    # Determine which coordinates (row/col) to explore based on mode
    if search_mode == "pos":
        rows, cols = np.where(real_part_data > threshold)
    elif search_mode == "neg":
        rows, cols = np.where(real_part_data < -threshold)
    else: # "abs" - explore both directions of significant change
        rows, cols = np.where(np.abs(real_part_data) > threshold)

    for r_idx, c_idx in zip(rows, cols):
        label_row = labels[r_idx]
        label_col = labels[c_idx]
        
        # Identify qubits that are active ('1') in these bitstrings
        qubits_row = [num_qubits - 1 - i for i, bit in enumerate(label_row) if bit == '1']
        qubits_col = [num_qubits - 1 - i for i, bit in enumerate(label_col) if bit == '1']
        
        # We handle the "Basal to Interacting" shift by creating pairs 
        # from all qubits involved in this specific peak.
        # create_cnot_pairs_from_locations handles the "back and forth" internally.
        if qubits_row or qubits_col:
            new_pairs = create_cnot_pairs_from_locations(qubits_row, qubits_col)
            all_cnot_configurations.update(new_pairs)

    cnot_candidates = sorted(list(all_cnot_configurations))
    
    # --- 5. Optional Plotting ---
    if plot_filename or show_plot:
        # Diverging map centered at 0 so gains (co>mono) and losses (co<mono) read
        # clearly as red vs blue; bitstring row/col labels so peaks are identifiable.
        fig, ax = plt.subplots(figsize=(8, 7))
        _fs = max(4, min(9, int(220 / max(len(labels), 1))))
        # gauge-free display: keep the normalized-vector values as-is, only
        # standardize the COLOR range symmetrically around 0 so it reads evenly.
        _vmax = float(np.max(np.abs(real_part_data))) or 1.0
        sns.heatmap(real_part_data, annot=False, cmap="RdBu_r", center=0,
                    vmin=-_vmax, vmax=_vmax, cbar=True, ax=ax,
                    xticklabels=labels, yticklabels=labels,
                    cbar_kws={"label": "co - mono (density-matrix difference)"})
        ax.tick_params(axis="x", labelsize=_fs, rotation=90)
        ax.tick_params(axis="y", labelsize=_fs, rotation=0)
        ax.set_xlabel("state |cell2 cell1>"); ax.set_ylabel("state |cell2 cell1>")
        ax.set_title(f"DM Difference (mode: {search_mode})  —  red = gained in co, blue = lost")
        plt.tight_layout()
        if plot_filename: plt.savefig(plot_filename, dpi=300)
        if show_plot: plt.show()
        plt.close(fig)

    # --- 6. Verbose Printing ---
    if verbose_print:
        total_possible = num_qubits * (num_qubits - 1) if num_qubits > 1 else 0
        print(f"\n--- CNOT Analysis ({search_mode}) ---")
        print(f"Max Signal: {stats['max_element']:.4f}")
        print(f"Candidates Found: {len(cnot_candidates)} / {total_possible}")
        print("------------------------------\n")
        
    return cnot_candidates, stats


# ==============================================================================
# --- SCALABLE / LOW-PRECISION VARIANTS (additive; the function above is the
#     reference implementation. These return the SAME (candidates, stats).) ---
# ==============================================================================
def _candidate_pairs_from_peaks(index_pairs, num_qubits):
    """Map (row_index, col_index) density-matrix peaks -> deduplicated CNOT pairs,
    using the identical logic as find_cnot_candidates_from_state_diff."""
    cfg = set()
    for r_idx, c_idx in index_pairs:
        lr = bin(r_idx)[2:].zfill(num_qubits)
        lc = bin(c_idx)[2:].zfill(num_qubits)
        qr = [num_qubits - 1 - i for i, b in enumerate(lr) if b == '1']
        qc = [num_qubits - 1 - i for i, b in enumerate(lc) if b == '1']
        if qr or qc:
            cfg.update(create_cnot_pairs_from_locations(qr, qc))
    return sorted(cfg)


def _amp_map(x, norm):
    """Sparse amplitude map {state_index: amplitude} for one register, plus its
    qubit count. Matches _normalize_amplitudes: 'l1' -> sqrt(v/sum(v));
    None/'l2' -> v/||v||_2. Works from a bitstring dict or a dense array without
    densifying anything larger than the input."""
    if isinstance(x, dict):
        if not x:
            return {}, 0
        n = len(next(iter(x)))
        idxs = np.array([int(bs, 2) for bs in x.keys()], dtype=np.int64)
        vals = np.array([float(v) for v in x.values()], dtype=np.float64)
    else:
        arr = np.asarray(x, dtype=np.float64)
        if arr.size == 0:
            return {}, 0
        n = int(round(np.log2(arr.size)))
        idxs = np.nonzero(arr)[0]
        vals = arr[idxs]
    if norm == "l1":
        tot = vals.sum()
        amps = np.sqrt(np.clip(vals / tot, 0.0, None)) if tot > 0 else vals
    else:  # None or 'l2'  (value / ||value||_2)
        nrm = np.linalg.norm(vals)
        amps = vals / nrm if nrm > 0 else vals
    return {int(i): float(a) for i, a in zip(idxs, amps)}, n


def _dm_stats_from_block(block, has_zeros, n_total, search_mode):
    """Reproduce find_cnot_candidates_from_state_diff's stats from the (possibly
    support-only) real difference block. has_zeros=True when off-block entries
    (all exactly 0) exist in the full 2^N x 2^N matrix."""
    mx = float(block.max()) if block.size else 0.0
    mn = float(block.min()) if block.size else 0.0
    if has_zeros:
        mx = max(mx, 0.0); mn = min(mn, 0.0)
    stats = {"max_element": mx, "min_element": mn,
             "abs_average": float(np.abs(block).sum()) / (2.0 ** (2 * n_total)) if n_total else 0.0}
    pos = block[block > 0]; neg = block[block < 0]
    if search_mode == "pos":
        stats["relevant_avg"] = float(pos.mean()) if pos.size else 0.0
    elif search_mode == "neg":
        stats["relevant_avg"] = float(neg.mean()) if neg.size else 0.0
    else:
        stats["avg_positive"] = float(pos.mean()) if pos.size else 0.0
        stats["avg_negative"] = float(neg.mean()) if neg.size else 0.0
    return stats


def find_cnot_candidates_dense(
    state_probs_initial1, state_probs_initial2,
    state_vec_probs_target1, state_vec_probs_target2,
    threshold, search_mode="pos", verbose_print=False, norm=None,
    precision="float32", plot_filename=None, show_plot=False,
):
    """Real-valued dense variant (no complex128 / DensityMatrix). Builds the full
    2^N x 2^N difference in the chosen float precision. Candidate-identical to
    find_cnot_candidates_from_state_diff; use for small N (allows the heatmap)."""
    dtype = {"float16": np.float16, "float32": np.float32, "float64": np.float64}[precision]
    v1i, _ = _process_target_state_input(state_probs_initial1, norm)
    v2i, _ = _process_target_state_input(state_probs_initial2, norm)
    v1t, _ = _process_target_state_input(state_vec_probs_target1, norm)
    v2t, _ = _process_target_state_input(state_vec_probs_target2, norm)
    a0 = np.kron(v2i, v1i).real.astype(dtype)
    a = np.kron(v2t, v1t).real.astype(dtype)
    if a0.size == 0 or a.size == 0:
        return [], {}
    D = (np.outer(a, a) - np.outer(a0, a0))
    n_total = int(round(np.log2(D.shape[0])))
    if search_mode == "pos":
        rows, cols = np.where(D > threshold)
    elif search_mode == "neg":
        rows, cols = np.where(D < -threshold)
    else:
        rows, cols = np.where(np.abs(D) > threshold)
    cands = _candidate_pairs_from_peaks(zip(rows.tolist(), cols.tolist()), n_total)
    stats = _dm_stats_from_block(D, has_zeros=False, n_total=n_total, search_mode=search_mode)
    if plot_filename or show_plot:
        _plot_dm_heatmap(np.asarray(D, dtype=np.float64),
                         [bin(i)[2:].zfill(n_total) for i in range(D.shape[0])],
                         search_mode, plot_filename, show_plot)
    if verbose_print:
        tp = n_total * (n_total - 1) if n_total > 1 else 0
        print(f"\n--- CNOT Analysis ({search_mode}, dense-{precision}) ---")
        print(f"Max Signal: {stats['max_element']:.4f}")
        print(f"Candidates Found: {len(cands)} / {tp}\n------------------------------\n")
    return cands, stats


def find_cnot_candidates_sparse(
    state_probs_initial1, state_probs_initial2,
    state_vec_probs_target1, state_vec_probs_target2,
    threshold, search_mode="pos", verbose_print=False, norm=None,
    precision="float32", plot_filename=None, show_plot=False, plot_support_cap=64,
):
    """Scalable support-block variant. Never builds the 2^N vector or 2^N x 2^N
    matrix: it works on the union support S of the (sparse) basal/target combined
    amplitudes and computes the |S| x |S| block only. Candidate-identical to the
    reference for threshold > 0. Memory scales with the DATA (|S|^2), not 4^N.
    Combined index convention matches np.kron(cell2, cell1): g = i2 * 2**n1 + i1.
    """
    m1i, n1 = _amp_map(state_probs_initial1, norm)
    m2i, n2 = _amp_map(state_probs_initial2, norm)
    m1t, n1t = _amp_map(state_vec_probs_target1, norm)
    m2t, n2b = _amp_map(state_vec_probs_target2, norm)
    n1 = n1 or n1t; n2 = n2 or n2b
    if not (m1t and m2t):
        return [], {}
    n_total = n1 + n2

    def _combine(m1, m2):
        out = {}
        base = 1 << n1
        for i2, a2 in m2.items():
            off = i2 * base
            for i1, a1 in m1.items():
                out[off + i1] = a2 * a1
        return out

    a_map = _combine(m1t, m2t)                 # target (co)
    a0_map = _combine(m1i, m2i) if (m1i and m2i) else {}   # basal (mono)
    S = sorted(set(a_map) | set(a0_map))
    # Threshold pruning (exact): an element a_r a_c - a0_r a0_c can only exceed
    # +threshold if |a_r| * max|a| > threshold (and the same for c), and can only
    # fall below -threshold if |a0_r| * max|a0| > threshold.  Dropping every other
    # index therefore removes no entry that passes the threshold, while the block
    # shrinks from |support|^2 to a few hundred squared at 20 qubits.
    if threshold > 0 and S:
        am = max((abs(v) for v in a_map.values()), default=0.0)
        a0m = max((abs(v) for v in a0_map.values()), default=0.0)
        keep_pos = search_mode in ("pos", "abs")
        keep_neg = search_mode in ("neg", "abs")
        S = [k for k in S
             if (keep_pos and abs(a_map.get(k, 0.0)) * am > threshold)
             or (keep_neg and abs(a0_map.get(k, 0.0)) * a0m > threshold)]
    dtype = {"float16": np.float16, "float32": np.float32, "float64": np.float64}[precision]
    aS = np.array([a_map.get(k, 0.0) for k in S], dtype=dtype)
    a0S = np.array([a0_map.get(k, 0.0) for k in S], dtype=dtype)
    block = np.outer(aS, aS) - np.outer(a0S, a0S)      # |S| x |S| only

    if search_mode == "pos":
        pr, pc = np.where(block > threshold)
    elif search_mode == "neg":
        pr, pc = np.where(block < -threshold)
    else:
        pr, pc = np.where(np.abs(block) > threshold)
    # map local support positions back to global state indices
    idx_pairs = ((S[p], S[q]) for p, q in zip(pr.tolist(), pc.tolist()))
    cands = _candidate_pairs_from_peaks(idx_pairs, n_total)

    has_zeros = len(S) < (1 << n_total)
    stats = _dm_stats_from_block(block, has_zeros=has_zeros, n_total=n_total, search_mode=search_mode)

    if (plot_filename or show_plot) and len(S) <= plot_support_cap:
        # heatmap of the support block only, labeled by the S states
        _plot_dm_heatmap(np.asarray(block, dtype=np.float64),
                         [bin(k)[2:].zfill(n_total) for k in S],
                         f"{search_mode}, support {len(S)}", plot_filename, show_plot)
    elif (plot_filename or show_plot):
        print(f"[find_cnot_candidates_sparse] support {len(S)} > cap {plot_support_cap}; heatmap skipped.")

    if verbose_print:
        tp = n_total * (n_total - 1) if n_total > 1 else 0
        print(f"\n--- CNOT Analysis ({search_mode}, sparse |S|={len(S)}) ---")
        print(f"Max Signal: {stats['max_element']:.4f}")
        print(f"Candidates Found: {len(cands)} / {tp}\n------------------------------\n")
    return cands, stats


def _plot_dm_heatmap(real_part_data, labels, search_mode, plot_filename=None, show_plot=False):
    """Shared DM-difference heatmap: diverging map centered at 0, symmetric color
    limits (gauge-free display), bitstring axis labels, 300 dpi PNG."""
    fig, ax = plt.subplots(figsize=(8, 7))
    _fs = max(4, min(9, int(220 / max(len(labels), 1))))
    _vmax = float(np.max(np.abs(real_part_data))) or 1.0
    sns.heatmap(real_part_data, annot=False, cmap="RdBu_r", center=0,
                vmin=-_vmax, vmax=_vmax, cbar=True, ax=ax,
                xticklabels=labels, yticklabels=labels,
                cbar_kws={"label": "co - mono (density-matrix difference)"})
    ax.tick_params(axis="x", labelsize=_fs, rotation=90)
    ax.tick_params(axis="y", labelsize=_fs, rotation=0)
    ax.set_xlabel("state |cell2 cell1>"); ax.set_ylabel("state |cell2 cell1>")
    ax.set_title(f"DM Difference (mode: {search_mode})  —  red = gained in co, blue = lost")
    plt.tight_layout()
    if plot_filename:
        plt.savefig(plot_filename, dpi=300)
    if show_plot:
        plt.show()
    plt.close(fig)


# ==============================================================================
# --- MAIN SEARCH ALGORITHMS (REFACTORED) ---
# ==============================================================================
def _run_single_greedy_search_from_start(
    circ1, circ2, state_vec_probs_target1, state_vec_probs_target2,
    all_possible_single_cnots, starting_cnot, min_cnot_depth, nshots, kl_tol,
    initial_kl_for_path, 
    max_cnot_depth=30  
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
    path_history = []  # Add this list
    path_history.append((initial_kl_for_path, [starting_cnot]))    # Record the very first CNOT added
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
                temp_circuit_with_cnots = apply_entanglement_topology(temp_base_circuit, ng_circ1, temp_cnot_sequence)
                
                kl_div1, kl_div2 = score_circuit_kl_divergences(temp_circuit_with_cnots, state_vec_probs_target1, state_vec_probs_target2, nshots)
                
                if kl_div1 is not None and kl_div2 is not None:
                    current_kl_sum = kl_div1 + kl_div2
                    
                    if current_kl_sum < current_iteration_min_kl_sum:
                        current_iteration_min_kl_sum = current_kl_sum
                        current_iteration_best_cnot = candidate_cnot
                        improvement_made = True
                        
        if improvement_made:
            best_cnot_sequence.append(current_iteration_best_cnot)
            current_kl_sum_on_path = current_iteration_min_kl_sum
            # RECORD EVERY STEP
            path_history.append((current_kl_sum_on_path, list(best_cnot_sequence)))

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
            
    return path_history, best_sequence_on_path, best_kl_on_path

def _run_greedy_removal_search(
    circ1, circ2, state_vec_probs_target1, state_vec_probs_target2,
    initial_cnot_sequence, min_kl_sum_initial, min_cnot_depth, nshots, kl_tol
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

    removal_history = []
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
            
            temp_circuit_with_cnots = apply_entanglement_topology(temp_base_circuit, ng_circ1, temp_cnot_sequence)
            kl_div1, kl_div2 = score_circuit_kl_divergences(temp_circuit_with_cnots, state_vec_probs_target1, state_vec_probs_target2, nshots)
            
            if kl_div1 is not None and kl_div2 is not None:
                current_kl_after_removal = kl_div1 + kl_div2
                
                # Check for absolute improvement over the current path score
                if current_kl_after_removal < (best_kl_after_removal - (kl_tol * 0.3) ):
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
            removal_history.append((current_kl_sum_on_path, list(best_cnot_sequence)))

            # Update the overall best sequence found so far
            if current_kl_sum_on_path < best_kl_on_path:
                best_kl_on_path = current_kl_sum_on_path
                best_sequence_on_path = list(best_cnot_sequence)
            
            print(f"  Removed CNOT {best_cnot_to_remove}. New KL Sum: {current_kl_sum_on_path:.6f} at depth {len(best_cnot_sequence)}")

        else:
            print("  No CNOT removal improved the score. Stopping removal search.")
            break
            
    return removal_history, best_sequence_on_path, best_kl_on_path


def find_best_cnot_sequence_multi_epoch(
    circ1: QuantumCircuit,
    circ2: QuantumCircuit,
    state_probs_initial1: dict,
    state_probs_initial2: dict,
    state_vec_probs_target1: dict,
    state_vec_probs_target2: dict,
    # --- NEW: Optional parameter to provide a candidate list ---
    cnot_search_candidates: Optional[List[Tuple[int, int]]] = None,
    n_epochs: Optional[int] = None,   # None / <=0 => maximum (one epoch per candidate)
    min_cnot_depth: int = 1,
    nshots: int = 1000,
    threshold: float = 0.1,
    search_mode: str = "abs", # abs, pos, neg
    kl_tol: float = 0.05,
    ratio_kl_tol: float = 0.6,
    max_greedy_depth: int = 30
):
    """
    Performs a multi-epoch greedy search with CNOT removal refinement.

    If `cnot_search_candidates` is provided, it uses that list for the search.
    Otherwise, it generates candidates by analyzing the density matrix difference.
    """
    ng_circ1 = circ1.num_qubits
    print(f"\n--- Starting Multi-Epoch Refined Search ---")
    start_total_time = time.time()

    # --- NEW LOGIC: Decide where to get CNOT candidates from ---
    if cnot_search_candidates is not None:
        print("\n--- Using provided CNOT candidate list for the search. ---")
        all_possible_single_cnots = cnot_search_candidates
        print(f"Number of candidates to be searched: {len(all_possible_single_cnots)}")
    else:
        print("\n--- Generating CNOT candidates from density matrix difference. ---")
        all_possible_single_cnots, stats = find_cnot_candidates_from_state_diff(
            state_probs_initial1, state_probs_initial2,
            state_vec_probs_target1, state_vec_probs_target2,
            threshold=threshold, search_mode=search_mode, verbose_print=True,
            plot_filename=None, show_plot=True
        )

    # --- The rest of the function proceeds as before ---
    base_combined_circuit = concatenate_circuits_with_separate_measurements(circ1, circ2)
    base_circuit_with_measurements = apply_entanglement_topology(base_combined_circuit, ng_circ1, [])
    kl_div1_no_cnot, kl_div2_no_cnot = score_circuit_kl_divergences(base_circuit_with_measurements, state_vec_probs_target1, state_vec_probs_target2, nshots)
    initial_kl_sum = kl_div1_no_cnot + kl_div2_no_cnot if kl_div1_no_cnot is not None and kl_div2_no_cnot is not None else float('inf')
    print(f"Initial KL divergence (baseline): {initial_kl_sum:.6f}")
    
    # 1. Initialize the Leaderboard (Record keeping)
    # Stores tuples of (kl_sum, sequence)
    evaluation_history = []
    
    # Record the baseline
    evaluation_history.append((initial_kl_sum, []))

    best_overall_sequence = []
    best_overall_kl_sum = initial_kl_sum

    if not all_possible_single_cnots:
        print("\nNo CNOT candidates found or provided. Skipping search epochs.")
    else:
        # --- Multi-Epoch greedy search (addition) ---
        # n_epochs=None (or <=0) means "maximum": one epoch per candidate, i.e.
        # try every candidate as a starting point (exhaustive over starts).
        if n_epochs is None or n_epochs <= 0:
            num_epochs_to_run = len(all_possible_single_cnots)
        else:
            num_epochs_to_run = min(n_epochs, len(all_possible_single_cnots))
        shuffled_candidates = all_possible_single_cnots.copy()
        random.Random(_GLOBAL_SEED).shuffle(shuffled_candidates)  # reproducible epoch-start order

        for epoch in range(num_epochs_to_run):
            starting_cnot = shuffled_candidates[epoch]

            temp_kl_scores = score_circuit_kl_divergences(
                apply_entanglement_topology(base_combined_circuit, ng_circ1, [starting_cnot]),
                state_vec_probs_target1, state_vec_probs_target2, nshots
            )
            if temp_kl_scores is not None and temp_kl_scores[0] is not None:
                temp_kl_sum = temp_kl_scores[0] + temp_kl_scores[1]
                print(f"\n--- Starting Epoch {epoch + 1}/{num_epochs_to_run} (Addition) with CNOT: {starting_cnot} (KL: {temp_kl_sum:.6f}) ---")
            else:
                print(f"\n--- Starting Epoch {epoch + 1}/{num_epochs_to_run} with CNOT: {starting_cnot} (KL calculation failed) ---")
                continue

            if temp_kl_sum >= initial_kl_sum:
                print(f"  Skipping epoch: Single CNOT KL ({temp_kl_sum:.6f}) does not improve over baseline ({initial_kl_sum:.6f}).")
                continue

            path_hist, best_sequence_this_epoch, min_kl_this_epoch = _run_single_greedy_search_from_start(
                circ1, circ2, state_vec_probs_target1, state_vec_probs_target2,
                all_possible_single_cnots, starting_cnot, min_cnot_depth, nshots, kl_tol,
                initial_kl_for_path=temp_kl_sum,
                max_cnot_depth=max_greedy_depth
            )
            evaluation_history.extend(path_hist)

            print(f"  Epoch {epoch + 1} best KL Sum: {min_kl_this_epoch:.6f}")
            if min_kl_this_epoch < best_overall_kl_sum: # Simplified check for direct improvement
                best_overall_kl_sum = min_kl_this_epoch
                best_overall_sequence = best_sequence_this_epoch
                print(f"  --> Epoch {epoch + 1} found a new overall best KL Sum: {best_overall_kl_sum:.6f}")            
                # --- Greedy CNOT removal search ---
                if best_overall_sequence:
                    print("\n--- Starting Greedy CNOT Removal Search on Best Found Sequence ---")
                    rem_hist, best_sequence_after_removal, best_kl_after_removal = _run_greedy_removal_search(
                        circ1, circ2, state_vec_probs_target1, state_vec_probs_target2,
                        best_overall_sequence, best_overall_kl_sum, 0, nshots, kl_tol = kl_tol
                    )
                    

                    evaluation_history.extend(rem_hist)

                    if best_kl_after_removal < best_overall_kl_sum: # Simplified check
                        best_overall_kl_sum = best_kl_after_removal
                        best_overall_sequence = best_sequence_after_removal
                        print(f"\n--> Removal search found a new overall best KL Sum: {best_overall_kl_sum:.6f}")

    # --- THE FINAL SELECTION LOGIC (Occam's Razor) ---
    if evaluation_history:
        # 1. Sort by number of CNOTs (fewest gates first)
        evaluation_history.sort(key=lambda x: len(x[1]))

        # 2. Start with the absolute simplest (usually the 0-CNOT baseline)
        final_best_sequence = evaluation_history[0][1]
        final_best_kl = evaluation_history[0][0]

        for kl_score, sequence in evaluation_history:
            # "Occam's Razor": Only move to a longer sequence if it 
            # provides a significant improvement (defined by kl_tol)
            if kl_score < (final_best_kl - kl_tol):
                final_best_kl = kl_score
                final_best_sequence = sequence
        
        best_overall_sequence = final_best_sequence
        best_overall_kl_sum = final_best_kl

    # Change this final check:
    if len(best_overall_sequence) == 0:
        print(f"\nFinal Result: Baseline (0 CNOTs) is the best choice (KL: {best_overall_kl_sum:.6f}).")
    else:
        print(f"\nFinal Result: Best sequence found has {len(best_overall_sequence)} CNOTs (KL: {best_overall_kl_sum:.6f}).")

    end_total_time = time.time()
    print(f"\nTotal search took: {end_total_time - start_total_time:.2f} seconds.")

    return best_overall_sequence, best_overall_kl_sum


def _single_cnot_insertion_search(
    base_circuit: 'QuantumCircuit',
    ng_circ1: int,
    current_sequence: list,
    candidate_cnots: list,
    target_probs1: dict,
    target_probs2: dict,
    nshots: int,
    kl_tol: float, 
    ratio_kl_tol: float,
    current_baseline_kl: float = float('inf') # Added this to anchor the search
):
    """
    Finds the best single CNOT to insert into the current sequence at the best location,
    requiring an improvement greater than kl_tol.
    """
    best_single_cnot = None
    best_insertion_index = -1
    
    # Anchor the search to the current best score
    min_kl_after_add = current_baseline_kl 
    n_trials = 0
    
    remaining_cnots = [c for c in candidate_cnots if c not in current_sequence]
    if not remaining_cnots:
        return current_sequence, current_baseline_kl, None

    for cnot_to_add in remaining_cnots:
        for i in range(len(current_sequence) + 1):
            n_trials += 1
            trial_sequence = current_sequence[:i] + [cnot_to_add] + current_sequence[i:]
            
            trial_circuit = apply_entanglement_topology(base_circuit, ng_circ1, trial_sequence)
            kl_divs = score_circuit_kl_divergences(trial_circuit, target_probs1, target_probs2, nshots)
            
            if kl_divs is not None:
                kl_sum = kl_divs[0] + kl_divs[1]
                
                # Use the absolute significance threshold
                if kl_sum < (min_kl_after_add - kl_tol):                   
                    min_kl_after_add = kl_sum
                    best_single_cnot = cnot_to_add
                    best_insertion_index = i
    
    if best_single_cnot is not None:
        new_sequence = current_sequence[:best_insertion_index] + [best_single_cnot] + current_sequence[best_insertion_index:]
        print(f"   - Single Insertion Phase: Added {best_single_cnot} at index {best_insertion_index}. New KL: {min_kl_after_add:.6f}")
        return new_sequence, min_kl_after_add, best_single_cnot
    
    return current_sequence, current_baseline_kl, None


def _single_cnot_deletion_search(
    base_circuit: 'QuantumCircuit',
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
    Finds the single CNOT whose removal yields the best KL divergence sum,
    accounting for simulation noise.
    """
    if not current_sequence:
        return current_sequence, float('inf'), None

    # 1. Establish the current baseline
    base_circuit_kl = apply_entanglement_topology(base_circuit, ng_circ1, current_sequence)
    kl_divs_base = score_circuit_kl_divergences(base_circuit_kl, target_probs1, target_probs2, nshots)
    base_kl_sum = kl_divs_base[0] + kl_divs_base[1] if kl_divs_base is not None else float('inf')
    
    # 2. Define a Slack/Epsilon 
    # If shots=1000, noise is ~0.03-0.05. 
    # We only delete if it significantly improves OR if we want to prune redundant gates.
    # To be conservative, let's look for any strictly better KL.
    min_kl_after_del = base_kl_sum 
    best_deletion_index = -1
    n_trials = 0

    for i in range(len(current_sequence)):
        n_trials += 1
        trial_sequence = current_sequence[:i] + current_sequence[i+1:]
        trial_circuit = apply_entanglement_topology(base_circuit, ng_circ1, trial_sequence)
        kl_divs = score_circuit_kl_divergences(trial_circuit, target_probs1, target_probs2, nshots)
        
        if kl_divs is not None:
            kl_sum = kl_divs[0] + kl_divs[1]
            
            # IMPROVEMENT CHECK:
            # We accept a deletion if the new KL is strictly better than our current best.
            # To avoid chasing noise (4.1 -> 4.09), you could use:
            if kl_sum < (min_kl_after_del - 0.5*kl_tol):
                min_kl_after_del = kl_sum
                best_deletion_index = i
                
    if best_deletion_index != -1:
        deleted_cnot = current_sequence[best_deletion_index]
        new_sequence = current_sequence[:best_deletion_index] + current_sequence[best_deletion_index+1:]
        print(f"   - Deletion Phase: Best Removed {deleted_cnot}. New KL: {min_kl_after_del:.6f}")
        return new_sequence, min_kl_after_del, deleted_cnot
    
    return current_sequence, base_kl_sum, None
    


def _pairwise_addition_search(
    base_circuit: 'QuantumCircuit',
    ng_circ1: int,
    initial_sequence: list,
    candidate_cnots: list,
    target_probs1: dict,
    target_probs2: dict,
    nshots: int,
    kl_tol: float = 0.05,
    nchoose: int = 2,
    ratio_kl_tol: float = 0.6
):
    print("\n  - Starting Efficiency-Optimized Search -")
    start_total = time.time()
    current_sequence = list(initial_sequence)
    
    # Establish baseline
    base_measure_circ = apply_entanglement_topology(base_circuit, ng_circ1, current_sequence)
    kl_divs_base = score_circuit_kl_divergences(base_measure_circ, target_probs1, target_probs2, nshots)
    best_kl_sum = kl_divs_base[0] + kl_divs_base[1] if kl_divs_base is not None else float('inf')
    
    improvement_made = True
    iteration = 0
    
    while improvement_made:
        iteration += 1
        # Reset at the start of each iteration
        improvement_made = False 
        
        print(f"\n--- Iteration {iteration} | Current KL: {best_kl_sum:.6f} ---")

        # 1. PHASE 1: SINGLE INSERTION
        t0 = time.time()
        new_seq_s, new_kl_s, added_cnot = _single_cnot_insertion_search(
            base_circuit, ng_circ1, current_sequence, candidate_cnots,
            target_probs1, target_probs2, nshots, kl_tol, ratio_kl_tol, 
            current_baseline_kl=best_kl_sum 
        )
        
        if added_cnot and new_kl_s < (best_kl_sum - kl_tol):
            current_sequence = new_seq_s
            best_kl_sum = new_kl_s
            improvement_made = True # This ensures the while loop runs again
            print(f"  [Phase 1] Added Single: {added_cnot} | New KL: {best_kl_sum:.6f}")

        # 2. PHASE 2: PAIRWISE (Runs on the potentially updated current_sequence)
        remaining_cnots = [c for c in candidate_cnots if c not in current_sequence]
        if len(remaining_cnots) >= nchoose:
            t0 = time.time()
            best_pair_seq = None
            pair_kl_best = best_kl_sum
            found_pair = None
            
            for pair in itertools.permutations(remaining_cnots, nchoose):
                trial_seq = current_sequence + list(pair)
                trial_circ = apply_entanglement_topology(base_circuit, ng_circ1, trial_seq)
                kl_divs = score_circuit_kl_divergences(trial_circ, target_probs1, target_probs2, nshots)
                
                if kl_divs:
                    kl_sum = kl_divs[0] + kl_divs[1]
                    # Must beat the current best (including potential Phase 1 update)
                    if kl_sum < (pair_kl_best - kl_tol):
                        pair_kl_best = kl_sum
                        best_pair_seq = trial_seq
                        found_pair = pair

            if found_pair:
                current_sequence = best_pair_seq
                best_kl_sum = pair_kl_best
                improvement_made = True # This keeps the search alive
                print(f"  [Phase 2] Added Pair: {found_pair} | New KL: {best_kl_sum:.6f}")

        # 3. PHASE 3: PRUNING (Clean up any redundancy)
        t0 = time.time()
        new_seq_d, new_kl_d, deleted_cnot = _single_cnot_deletion_search(
            base_circuit, ng_circ1, current_sequence, candidate_cnots,
            target_probs1, target_probs2, nshots, kl_tol, ratio_kl_tol
        )
        
        if deleted_cnot and new_kl_d < (best_kl_sum - 0.0001): # Use smaller epsilon for pruning
            current_sequence = new_seq_d
            best_kl_sum = new_kl_d
            improvement_made = True # Even pruning keeps the search alive to try adding more
            print(f"  [Phase 3] Pruned: {deleted_cnot} | New KL: {best_kl_sum:.6f}")

        # Check Global Target
        if best_kl_sum < kl_tol:
            print("  - Global Target KL met. Stopping.")
            break
            
        if not improvement_made:
            print("  - No improvements in any phase. Search Terminated.")

    print(f"\n--- Total Search Time: {time.time() - start_total:.2f}s ---")
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
    search_mode: str = "abs",  # "abs", "pos", or "neg"
    nchoose: int = 2,
    kl_tol: float = 0.05, 
    ratio_kl_tol: float = 0.6,
    cnot_search_candidates: list = None,
    ):
    """
    Performs an iterative optimization loop by separating pairwise addition
    and removal phases.

    cnot_search_candidates: optional pre-screened (and scoped) candidate list.
    If given, density-matrix screening is skipped and only these candidates are
    searched, so GRN-only / LR-only / GRN+LR scopes apply to the n-wise search
    exactly as they do to the multi-epoch search.
    """
    
    ng_circ1 = circ1.num_qubits

    print(f"\n--- Starting Iterative Pairwise Search ---")
    start_total_time = time.time()

    # --- State and Density Matrix Analysis ---
    if cnot_search_candidates is not None:
        initial_cnot_config = [tuple(c) for c in cnot_search_candidates]
    else:
        initial_cnot_config, stats = find_cnot_candidates_from_state_diff(
            state_probs_initial1, state_probs_initial2,
            state_vec_probs_target1, state_vec_probs_target2, 
            threshold= threshold, search_mode = search_mode, verbose_print = True,
            plot_filename = None, show_plot = True
            )

    print(f"Number of initial CNOT candidates: {len(initial_cnot_config)}")

    # --- Step 2: Baseline (no CNOTs) ---
    base_combined_circuit = concatenate_circuits_with_separate_measurements(circ1, circ2)
    base_circuit = apply_entanglement_topology(base_combined_circuit, ng_circ1, [])
    kl_divs = score_circuit_kl_divergences(base_circuit, state_vec_probs_target1, state_vec_probs_target2, nshots)
    initial_kl_sum = kl_divs[0] + kl_divs[1] if kl_divs is not None else float('inf')

    print(f"Initial KL-divergence: {initial_kl_sum:.6f}")

    if len(initial_cnot_config) < 2:
        print("Not enough CNOT candidates to form a pair. Skipping search.")
        # Return a 3-tuple to match the normal return path
        # (initial_cnot_config, best_add_sequence, best_add_kl_sum).
        return initial_cnot_config, [], initial_kl_sum

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



def build_kl_divergence_matrix_interaction(
    circ1: QuantumCircuit,
    circ2: QuantumCircuit,
    state_probs_initial1: dict,
    state_probs_initial2: dict,
    state_vec_probs_target1: dict,
    state_vec_probs_target2: dict,
    nshots: int = 1000,
    threshold: float = 0.05,
    search_mode: str = "abs", # abs, pos, neg
    include_single_cnot_kl: bool = True,
    cnot_search_candidates: list = None,
):
    """
    Builds a matrix where M[i][j] represents the KL divergence of the circuit
    formed by cnot_i followed by cnot_j.
    
    This function first identifies potential CNOT candidates by analyzing
    the density matrix difference between the initial and target states,
    and then calculates the KL divergence for single CNOTs and all CNOT pairs.

    Args:
        circ1 (QuantumCircuit): The first base quantum circuit.
        circ2 (QuantumCircuit): The second base quantum circuit.
        state_probs_initial1 (dict): Initial state probabilities for circ1.
        state_probs_initial2 (dict): Initial state probabilities for circ2.
        state_vec_probs_target1 (dict): Target state probabilities for circ1's part.
        state_vec_probs_target2 (dict): Target state probabilities for circ2's part.
        nshots (int): Number of shots for circuit simulation.
        threshold (float): Threshold to identify significant off-diagonal elements.
        include_single_cnot_kl (bool): If True, the diagonal M[i][i] will store
                                       the KL divergence of the circuit with only cnot_i.

    Returns:
        np.ndarray: A square matrix where M[i][j] is the KL divergence for
                    the sequence [cnot_i, cnot_j].
        dict: A mapping from CNOT tuple to its index in the matrix.
        list: The list of initial CNOT candidates used.
        float: The initial KL divergence for the circuit with no CNOTs.
    """
    ng_circ1 = circ1.num_qubits
    ng_circ2 = circ2.num_qubits
    
    print(f"\n--- Identifying Potential CNOT Linkers from Density Matrix Difference ---")


    # --- State and Density Matrix Analysis ---
    # (a pre-screened / scoped candidate list skips the screening, so the
    #  GRN / LR / GRN+LR scopes also apply to the VQE-hybrid search)
    if cnot_search_candidates is not None:
        initial_cnot_config = [tuple(c) for c in cnot_search_candidates]
    else:
        initial_cnot_config, stats = find_cnot_candidates_from_state_diff(
            state_probs_initial1, state_probs_initial2,
            state_vec_probs_target1, state_vec_probs_target2, 
            threshold= threshold, search_mode=search_mode, verbose_print = True,
            plot_filename = None, show_plot = True
            )

    # --- Calculate Baseline KL Divergence (no CNOTs) ---
    base_combined_circuit = concatenate_circuits_with_separate_measurements(circ1, circ2)
    base_circuit_no_cnots = apply_entanglement_topology(base_combined_circuit, ng_circ1, [])
    kl_divs_baseline = score_circuit_kl_divergences(base_circuit_no_cnots, state_vec_probs_target1, state_vec_probs_target2, nshots)
    initial_kl_sum = kl_divs_baseline[0] + kl_divs_baseline[1] if kl_divs_baseline is not None else float('inf')
    print(f"Initial KL Divergence (no CNOTs): {initial_kl_sum:.6f}")

    # --- Start building the matrix with the identified candidates ---
    num_candidates = len(initial_cnot_config)
    kl_divergence_matrix = np.full((num_candidates, num_candidates), np.inf)
    cnot_to_index = {tuple(cnot): i for i, cnot in enumerate(initial_cnot_config)}

    print(f"\n--- Building KL Divergence Matrix ({num_candidates}x{num_candidates}) ---")

    if include_single_cnot_kl:
        print("Calculating KL for single CNOTs (diagonal elements)...")
        for i, cnot_i in enumerate(initial_cnot_config):
            trial_circuit = apply_entanglement_topology(
                base_combined_circuit, ng_circ1, [cnot_i]
            )
            kl_divs = score_circuit_kl_divergences(
                trial_circuit, state_vec_probs_target1, state_vec_probs_target2, nshots
            )
            if kl_divs is not None:
                kl_divergence_matrix[i, i] = kl_divs[0] + kl_divs[1]
                # print(f"  KL for [{cnot_i}]: {kl_divergence_matrix[i, i]:.6f}")

    print("Calculating KL for CNOT pairs (off-diagonal elements)...")
    n_pairs_tested = 0
    for cnot_i_val in initial_cnot_config:
        for cnot_j_val in initial_cnot_config:
            idx_i = cnot_to_index[tuple(cnot_i_val)]
            idx_j = cnot_to_index[tuple(cnot_j_val)]

            if idx_i == idx_j:
                continue

            n_pairs_tested += 1
            trial_sequence = [cnot_i_val, cnot_j_val]
            trial_circuit = apply_entanglement_topology(
                base_combined_circuit, ng_circ1, trial_sequence
            )
            kl_divs = score_circuit_kl_divergences(
                trial_circuit, state_vec_probs_target1, state_vec_probs_target2, nshots
            )
            if kl_divs is not None:
                kl_divergence_matrix[idx_i, idx_j] = kl_divs[0] + kl_divs[1]
                # print(f"  KL for [{cnot_i_val}, {cnot_j_val}]: {kl_divergence_matrix[idx_i, idx_j]:.6f}")
    
    print(f"Total CNOT pairs tested: {n_pairs_tested}")
    print("--- KL Divergence Matrix Built ---")

    return kl_divergence_matrix, cnot_to_index, initial_cnot_config, initial_kl_sum


def kl_to_qubo_matrix(kl_matrix: np.ndarray, initial_kl_sum: float) -> np.ndarray:
    """
    Builds a QUBO matrix from a KL divergence matrix.

    This function transforms a matrix of KL divergences into a Quadratic Unconstrained
    Binary Optimization (QUBO) matrix. The resulting matrix can be used by a
    QUBO solver to find the optimal selection of CNOT gates that minimizes
    the KL divergence of a quantum circuit.

    The QUBO problem is formulated to minimize a cost function where the energy
    of a solution (a set of selected CNOTs) corresponds to the circuit's
    relative KL divergence.

    Args:
        kl_matrix (np.ndarray): A square matrix where:
                                - kl_matrix[k, k] is the KL divergence of a circuit with a
                                  single CNOT gate `k`.
                                - kl_matrix[k, l] is the KL divergence of a circuit with
                                  the sequence of CNOT gates `k` then `l`.
        initial_kl_sum (float): The baseline KL divergence of the circuit with
                                no CNOT gates. This value is used to calculate
                                the cost relative to the zero-CNOT circuit.

    Returns:
        np.ndarray: The resulting QUBO matrix, Q. This is an upper triangular
                    matrix (with the diagonal included) containing the coefficients
                    for the QUBO objective function:
                    E(x) = sum(Q_ij * x_i * x_j) for i <= j.
                    
    Raises:
        ValueError: If kl_matrix is not a square matrix.
    """
    if kl_matrix.shape[0] != kl_matrix.shape[1]:
        raise ValueError("kl_matrix must be a square matrix.")

    num_vars = kl_matrix.shape[0]
    qubo_matrix = np.zeros((num_vars, num_vars))

    # A large penalty is assigned to invalid or infinite KL divergences to
    # discourage the solver from selecting those CNOT gates or pairs.
    INF_PENALTY = 1e9

    # --- Step 1: Calculate Linear Coefficients (Diagonal Terms, Q_ii) ---
    # These terms represent the cost of selecting a single CNOT gate.
    for i in range(num_vars):
        if not np.isinf(kl_matrix[i, i]):
            # The linear cost is the single-gate KL divergence relative to the baseline.
            cost_i = kl_matrix[i, i] - initial_kl_sum
            qubo_matrix[i, i] = cost_i
        else:
            # Assign a large penalty for an infinite KL divergence.
            qubo_matrix[i, i] = INF_PENALTY

    # --- Step 2: Calculate Quadratic Coefficients (Off-Diagonal Terms, Q_ij) ---
    # These terms represent the synergistic interaction cost of selecting a pair
    # of CNOT gates, isolated from their individual contributions.
    for i in range(num_vars):
        for j in range(i + 1, num_vars):
            kl_ij = kl_matrix[i, j]
            kl_ji = kl_matrix[j, i]

            if not np.isinf(kl_ij) and not np.isinf(kl_ji):
                # We must choose *one* order to encode in Q_ij.
                # The lowest KL divergence of the two orders is the target cost.
                kl_best_order = min(kl_matrix[i, j], kl_matrix[j, i])
                
                # Get the pre-computed linear costs from the diagonal of the QUBO matrix.
                cost_i = qubo_matrix[i, i]
                cost_j = qubo_matrix[j, j]
                
                # Isolate the interaction cost using the BEST of the two orders.
                # This simplifies the optimization: if the solver chooses i and j, 
                # the cost will reflect the best physical sequence.
                interaction_cost = (kl_best_order - initial_kl_sum) - cost_i - cost_j
                qubo_matrix[i, j] = interaction_cost
                
            else:
                # Assign a large penalty if the KL divergence for the pair is infinite.
                qubo_matrix[i, j] = INF_PENALTY

    return qubo_matrix

#from qiskit.primitives import StatevectorEstimator

def vqe_solver(
    ansatz, # Renamed from 'cirquit' for common convention
    hamiltonian,
    backend,
    optimizer_method: str = "COBYLA", # "L-BFGS-B | COBYLA
    niter: int = 100
):
    """
    Performs a Variational Quantum Eigensolver (VQE) optimization.

    This function encapsulates the VQE workflow, including:
    1. Creating an interaction observable from histogram data.
    2. Setting up static and variable parameters for the quantum circuit.
    3. Initializing the Qiskit StatevectorEstimator.
    4. Running the optimization using scipy.optimize.minimize.
    5. Collecting cost function values during optimization.
    6. Updating parameters with optimized values.
    7. (Optional) Plotting the energy minimization curve.

    Args:
        circuit (QuantumCircuit): The parameterized quantum circuit (ansatz).
        act_percentages (list): Initial percentage values for parameters
                                containing '_act_' in the circuit.
        cost_func_wrapper (callable): The cost function to be minimized.
                                      It MUST accept arguments in the following order:
                                      cost_func_wrapper(current_variable_params_array,
                                                        static_params_dict,
                                                        circuit,
                                                        observable,
                                                        estimator,
                                                        variable_param_objects_list)
                                      Inside this function, you should combine
                                      static_params_dict and map current_variable_params_array
                                      to variable_param_objects_list to form the
                                      full parameter dictionary for circuit binding.
        min_ones_for_observable (int): Minimum number of '1's for filtering
                                       histogram data when creating the observable.
        optimizer_method (str): The optimization method to use for scipy.minimize
                                (e.g., "L-BFGS-B", "COBYLA", "SLSQP").

    Returns:
        tuple: A tuple containing:
            - result_object (OptimizeResult): The result object from scipy.optimize.minimize.
            - optimized_full_params (dict): The dictionary of all parameters
                                            with their final optimized values.
            - cost_history (list): A list of cost function values recorded
                                   during each optimization iteration.
    """
    # 1. Define the base options dictionary
    options = {
        "disp": False, # Optional: Set to True to display solver messages
        "tol": 1e-4,   # Optional: Set the tolerance for termination
        "maxiter":niter
    }

    num_qubits = ansatz.num_qubits # Get number of qubits from the circuit

    # make quantum circuit compatible to the backend
    pm = generate_preset_pass_manager(backend = backend, optimization_level=3)
    ansatz_isa = pm.run(ansatz)

    estimator = Estimator(mode=backend)
    estimator.options.default_shots = 1024

    # 4. Prepare initial guess for optimization initialize 0 vector
    num_params = ansatz.num_parameters 
    #x0_interaction = np.random.rand(num_params) * 2 * np.pi
    #x0_interaction = np.zeros(num_params)
    x0_interaction = np.ones(num_params)*np.pi/2
    #x0_interaction = np.ones(num_params)*np.pi
    
    # in the context of the vqe_solver function scope.
    iteration_data = {'counter': 0} 
    cost_values = [] # List to store cost values at each iteration
    
    def cost_func_vqe(params, ansatz, hamiltonian, estimator):
        """Return estimate of energy from estimator
        Parameters:
            params (ndarray): Array of ansatz parameters
            ansatz (QuantumCircuit): Parameterized ansatz circuit
            hamiltonian (SparsePauliOp): Operator representation of Hamiltonian
            estimator (EstimatorV2): Estimator primitive instance
            cost_history_dict: Dictionary for storing intermediate results
    
        Returns:
            float: Energy estimate
        """
        pub = (ansatz, [hamiltonian], [params])
        result = estimator.run(pubs=[pub]).result()
        energy = result[0].data.evs[0]

        return energy
 
    def cost_func_wrapper(xk, ansatz, hamiltonian, estimator):
        return  cost_func_vqe(xk, ansatz, hamiltonian, estimator) # Pass combined_qc

    # Define the callback function for minimize
    def callback_func(xk):
        # Access the counter from the enclosing scope's dictionary
        current_counter = iteration_data['counter']
        if current_counter > 100:
            print_criteria = 100
        else:
            print_criteria = 20

        current_cost = cost_func_wrapper(xk, ansatz, hamiltonian, estimator)
        cost_values.append(current_cost)

        # Print the current cost only every 20 iterations
        if current_counter % print_criteria == 0 or current_counter == 0:
            print(f"Iteration {current_counter}: Current cost: {current_cost}")
            
        iteration_data['counter'] += 1 # Increment the counter

    # 6. Call minimize with args
    print(f"Starting optimization with method: {optimizer_method}")
    result_interaction = minimize(
        cost_func_wrapper,
        x0_interaction,
        # IMPORTANT: Pass static_params and variable_param_objects as fixed arguments
        # The cost_func_wrapper will use xk (the first argument) with these to bind parameters.
        args=(ansatz_isa, hamiltonian, estimator),
        method=optimizer_method, # Use the passed optimizer_method
        callback=callback_func, # Use the defined callback function
        options=options,
    )

    print("\nOptimization Result:")
    print(result_interaction)
    print(f"\nFinal Energy: {result_interaction.fun}")

    # Update the full parameter dictionary with optimized variable parameters
    opt_values = result_interaction.x # opt_values is the numpy.ndarray
       
    opt_params_dict = dict(zip(ansatz.parameters, opt_values))
    # The keys of this dictionary are the symbolic parameter objects

    print("\nOptimized Full Parameters:")
    for param, value in opt_params_dict.items():
        # Now 'param' is the symbolic object with a '.name' attribute
        print(f"  {param.name}: {value}")

    # Return the results
    return result_interaction, opt_params_dict, cost_values

def evaluate_and_plot_ansatz(
    # Use a '*' to make all subsequent arguments keyword-only
    *,
    ansatz: QuantumCircuit,
    params: List[float],
    backend: Backend,
    shots: int = 1024,
    title: str = "VQE Quantum Sampler Results",
    figsize: Tuple[int, int] = (7, 5),
    show_plot: bool = True,
    filename: Optional[str] = None
) -> Tuple[Dict[str, int], QuantumCircuit]:
    """
    Evaluates a quantum ansatz, optionally plots results, and returns the circuit and counts.

    Args:
        ansatz (QuantumCircuit): The quantum circuit to evaluate.
        params (List[float]): The parameters to assign to the ansatz.
        backend (Backend): The Qiskit backend (or simulator) to run on.
        shots (int, optional): The number of shots for the measurement. Defaults to 1024.
        title (str, optional): The title for the histogram plot.
        figsize (Tuple[int, int], optional): The size of the plot figure.
        show_plot (bool, optional): If True, displays the plot. Defaults to True.
        filename (Optional[str], optional): If provided, saves the plot to this file.

    Returns:
        Tuple[Dict[str, int], QuantumCircuit]: A tuple containing:
            - The dictionary of measurement counts.
            - The bound and transpiled quantum circuit that was executed.
            
    Raises:
        Exception: Propagates any exception that occurs during circuit execution,
                   providing a more direct and informative error message.
    """
    # This 'try...except' is now cleaner. It lets errors propagate naturally.
    # We remove 'return None' because it hides the true source of the error.
    
    sampler = Sampler(mode=backend)
    bound_circuit = ansatz.copy()
    bound_circuit.assign_parameters(params, inplace=True)
    bound_circuit.measure_all()

    # Transpile the circuit for the specific backend
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    transpiled_circuit = pm.run(bound_circuit)

    # Run the job
    job = sampler.run([transpiled_circuit], shots=shots)
    pub_result = job.result()[0]
    counts = pub_result.data.meas.get_counts()

    # Plotting is now optional and controlled by 'show_plot'
    if show_plot or filename:
        sorted_counts = dict(sorted(counts.items()))
        x_labels = list(sorted_counts.keys())
        y_values = list(sorted_counts.values())

        plt.figure(figsize=figsize)
        plt.bar(x_labels, y_values)
        plt.xlabel("Measurement Outcomes")
        plt.ylabel("Counts")
        plt.title(title)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, bbox_inches='tight', dpi=300)
        
        if show_plot:
            plt.show()
            
        plt.close() # Close the figure to free memory after showing/saving

    return counts, transpiled_circuit

# ==============================================================================
# PREREQUISITES: Define the necessary helper functions
# These must be defined in your notebook cell before you can use them.
# ==============================================================================

# Assuming these functions are already defined in your environment as provided.
# If not, make sure to include them before running the main block.
def get_top_k_counts_little_endian(counts: dict, k: int = 1) -> list[str]:
    """Extracts top k outcome strings from counts and converts to little-endian."""
    sorted_items = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    top_k_strings_big_endian = [item[0] for item in sorted_items[:k]]
    return [s[::-1] for s in top_k_strings_big_endian]

def get_top_k_boolean_vectors(counts: dict, k: int = 1) -> list[list[bool]]:
    """Extracts top k solutions from counts as boolean vectors."""
    top_k_strings = get_top_k_counts_little_endian(counts, k)
    return [[char == '1' for char in s] for s in top_k_strings]

def map_vqe_solution_to_cnots(
    boolean_vector: list[bool], 
    cnot_candidates: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Maps a boolean solution vector back to the CNOT gates it represents."""
    return [cnot for is_selected, cnot in zip(boolean_vector, cnot_candidates) if is_selected]

# ==============================================================================
# --- VQE HYBRID SEARCH: Iterating through Top-k VQE Solutions ---
# This block will run the search for k=3, 2, and 1, and find the best overall.
# ==============================================================================

# Make sure these helper functions are defined in your notebook
# get_top_k_boolean_vectors, map_vqe_solution_to_cnots, etc.

def run_vqe_hybrid_search(
    vqe_counts: Dict[str, int],
    initial_cnot_config: List[Tuple[int, int]],
    circ1: QuantumCircuit,
    circ2: QuantumCircuit,
    state_probs_initial1: dict,
    state_probs_initial2: dict,
    state_vec_probs_target1: dict,
    state_vec_probs_target2: dict,
    num_solutions_to_test: int = 3,
    min_cnot_depth: int = 1,
    nshots: int = 1000,
    threshold: float = 0.1,
    search_mode: str = "abs", # abs, pos, neg
    kl_tol: float = 0.01,
    ratio_kl_tol: float = 0.6,
    max_greedy_depth: int = 30
) -> Tuple[Optional[List[Tuple[int, int]]], float, Optional[int]]:
    """
    Performs a full hybrid VQE-classical search to find the best CNOT sequence.

    This function takes the results from a VQE run (counts) and tests the top N
    solutions. It processes them in reverse order (e.g., 3rd best, then 2nd, then 1st)
    to prioritize potentially simpler, high-performing solutions first. For each VQE
    solution, it runs a multi-epoch greedy search and tracks the overall best result.

    Args:
        vqe_counts (Dict[str, int]): The measurement counts from the VQE run.
        initial_cnot_config (List[Tuple[int, int]]): The full list of CNOTs the VQE
                                                     was optimizing over.
        circ1, circ2 (QuantumCircuit): The base quantum circuits.
        state_probs_initial1, state_probs_initial2 (dict): Initial state probabilities.
        state_vec_probs_target1, state_vec_probs_target2 (dict): Target state probabilities.
        num_solutions_to_test (int): The number of top VQE solutions to check (e.g., 3).
        min_cnot_depth, nshots, threshold, kl_tol, ratio_kl_tol, max_greedy_depth:
            Parameters to be passed directly to the internal `find_best_cnot_sequence_multi_epoch`
            greedy search function.

    Returns:
        Tuple containing:
        - Optional[List[Tuple[int, int]]]: The best CNOT sequence found. None if no
          improvement was made over the baseline.
        - float: The minimum combined KL divergence achieved.
        - Optional[int]: The rank of the VQE solution that produced the best result.
          None if no solution was better than the baseline.
    """
    # --- Initialization for tracking the best result ---
    overall_best_kl_sum = float('inf')
    overall_best_cnot_sequence = None
    overall_best_rank = None
    found_an_improvement = False

    print("="*60)
    print(f"--- Preparing to test Top {num_solutions_to_test} VQE solutions in REVERSE order ---")
    print("="*60)
    
    top_k_solutions = get_top_k_boolean_vectors(vqe_counts, k=num_solutions_to_test)

    if not top_k_solutions:
        print("No VQE solutions found in the counts dictionary. Halting search.")
        # Calculate baseline KL to return something meaningful
        base_circuit = apply_entanglement_topology(concatenate_circuits_with_separate_measurements(circ1, circ2), circ1.num_qubits, [])
        kl1, kl2 = score_circuit_kl_divergences(base_circuit, state_vec_probs_target1, state_vec_probs_target2, nshots)
        baseline_kl = kl1 + kl2 if kl1 is not None else float('inf')
        return None, baseline_kl, None

    # --- Loop through the solutions in REVERSE order ---
    for i, solution_vector in reversed(list(enumerate(top_k_solutions))):
        rank = i + 1
        print(f"\n--- Testing VQE Solution (Rank {rank}/{len(top_k_solutions)}) ---")

        vqe_selected_candidates = map_vqe_solution_to_cnots(solution_vector, initial_cnot_config)
        
        print(f"This VQE Solution corresponds to a candidate subset of {len(vqe_selected_candidates)} CNOTs:")
        if vqe_selected_candidates:
            for cnot in vqe_selected_candidates:
                print(f"  q[{cnot[0]}] -> q[{cnot[1]}]")
        else:
            print("  (No CNOTs were selected). Skipping greedy search.")
            continue

        num_epochs_for_this_run = len(vqe_selected_candidates)
        print(f"\nSetting number of greedy search epochs to {num_epochs_for_this_run} (one for each candidate).")

        cnot_sequence_from_greedy, kl_sum_from_greedy = find_best_cnot_sequence_multi_epoch(
            circ1=circ1,
            circ2=circ2,
            state_probs_initial1=state_probs_initial1,
            state_probs_initial2=state_probs_initial2,
            state_vec_probs_target1=state_vec_probs_target1,
            state_vec_probs_target2=state_vec_probs_target2,
            cnot_search_candidates=vqe_selected_candidates,
            n_epochs=num_epochs_for_this_run, # Dynamically set
            min_cnot_depth=min_cnot_depth,
            nshots=nshots,
            threshold=threshold,
            search_mode=search_mode,
            kl_tol=kl_tol,
            ratio_kl_tol=ratio_kl_tol,
            max_greedy_depth=max_greedy_depth
        )
        
        print(f"\nResult for (Rank {rank}): Final KL Sum = {kl_sum_from_greedy:.6f}")
        if kl_sum_from_greedy < overall_best_kl_sum:
            print(f"  >>> NEW OVERALL BEST SOLUTION FOUND! <<<")
            overall_best_kl_sum = kl_sum_from_greedy
            overall_best_cnot_sequence = cnot_sequence_from_greedy
            overall_best_rank = rank
            found_an_improvement = True
            print(f"  New best KL sum: {overall_best_kl_sum:.6f}")
            print(f"  Origin: VQE solution at rank {overall_best_rank}")
        else:
            print(f"  This result ({kl_sum_from_greedy:.6f}) did not beat the current best ({overall_best_kl_sum:.6f}).")

    return overall_best_cnot_sequence, overall_best_kl_sum, overall_best_rank



def analyze_and_summarize_network(
    circ1: QuantumCircuit,
    circ2: QuantumCircuit,
    topology: list[tuple[int, int]],
    angles: list[float],
    state_vec_probs_target1: dict,
    state_vec_probs_target2: dict,
    gene_list: list[str],
    nshots: int = 5000
) -> tuple[pd.DataFrame, dict, set]:
    """
    Performs a comprehensive analysis of an optimized circuit topology.
    
    1.  Conducts an ablation study to quantify the contribution of each gate.
    2.  Summarizes the network structure by identifying interaction hubs.
    3.  Returns all results in structured formats (DataFrame, dict, set).

    Args:
        circ1, circ2: The initial state circuits.
        topology: The final, ordered (control, target) topology.
        angles: The final, optimized CRX angles for the topology.
        state_vec_probs_target1/2: The target distributions.
        gene_list: The master list of gene names, ordered by qubit index.
        nshots: The number of shots for simulation.

    Returns:
        A tuple containing:
        - pd.DataFrame: A DataFrame detailing the contribution of each gate.
        - dict: A dictionary summarizing interactions grouped by "hub" (control) gene.
        - set: A set of all unique gene names involved in the discovered network.
    """
    if not topology:
        print("No topology provided for analysis.")
        return pd.DataFrame(), {}, set()

    print("\n" + "="*70)
    print("--- Comprehensive Interaction Network Analysis & Summary ---")
    print("="*70)
    
    base_combined_circuit = concatenate_circuits_with_separate_measurements(circ1, circ2)
    
    # --- Step 1: Calculate Baseline KL Divergence ---
    baseline_circuit = add_crx_gates_and_measurements_to_circuit(base_combined_circuit, circ1.num_qubits, [], [])
    kl_divs_baseline = score_circuit_kl_divergences(baseline_circuit, state_vec_probs_target1, state_vec_probs_target2, nshots)
    current_kl = sum(kl_divs_baseline) if kl_divs_baseline and None not in kl_divs_baseline else float('inf')

    # --- Step 2: Perform Ablation Study ---
    # Two complementary views per gate:
    #   * Cumulative: KL as gates are added in order (order-dependent).
    #   * Isolated  : this gate ALONE vs baseline (order-independent, non-cumulative).
    baseline_kl_val = current_kl
    contribution_data = []
    contribution_data.append({
        "Step": 0, "Interaction": "Baseline (0 gates)",
        "CRX Angle (rad)": np.nan, "KL Divergence": current_kl,
        "KL Change": 0.0, "Cumulative KL Reduction": 0.0,
        "Isolated KL": np.nan, "Isolated Reduction": 0.0,
    })

    for i, (gate_topology, gate_angle) in enumerate(zip(topology, angles)):
        step_num = i + 1
        current_sequence_topology = topology[:step_num]
        current_sequence_angles = angles[:step_num]

        trial_circuit = add_crx_gates_and_measurements_to_circuit(
            base_combined_circuit, circ1.num_qubits, current_sequence_topology, current_sequence_angles
        )
        kl_divs = score_circuit_kl_divergences(trial_circuit, state_vec_probs_target1, state_vec_probs_target2, nshots)

        # Isolated (non-cumulative) contribution: circuit with ONLY this gate.
        iso_circuit = add_crx_gates_and_measurements_to_circuit(
            base_combined_circuit, circ1.num_qubits, [gate_topology], [gate_angle]
        )
        iso_divs = score_circuit_kl_divergences(iso_circuit, state_vec_probs_target1, state_vec_probs_target2, nshots)
        iso_kl = sum(iso_divs) if (iso_divs and None not in iso_divs) else float("nan")

        if kl_divs and None not in kl_divs:
            new_kl = sum(kl_divs)
            kl_change = new_kl - current_kl

            control_gene = gene_list[gate_topology[0]]
            target_gene = gene_list[gate_topology[1]]

            contribution_data.append({
                "Step": step_num,
                "Interaction": f"{control_gene} -> {target_gene}",
                "CRX Angle (rad)": gate_angle,
                "KL Divergence": new_kl,
                "KL Change": kl_change,
                "Cumulative KL Reduction": new_kl - contribution_data[0]["KL Divergence"],
                "Isolated KL": iso_kl,
                "Isolated Reduction": baseline_kl_val - iso_kl,   # >0 => this gate alone lowers KL
            })
            current_kl = new_kl
        else:
            print(f"Warning: Could not score circuit at step {step_num}. Skipping.")

    # --- Step 3: Create and Format the Final DataFrame ---
    contribution_df = pd.DataFrame(contribution_data)
    
    # NEW LOGIC: Use Baseline KL as the denominator for scientific grounding
    baseline_kl = contribution_df.iloc[0]["KL Divergence"]
    
    if baseline_kl > 1e-9:
        # % of baseline explained by this gate's cumulative step (order-dependent)
        contribution_df["% Contribution"] = (abs(contribution_df["KL Change"]) / baseline_kl) * 100
        # % of baseline explained by this gate ALONE (isolated, non-cumulative)
        contribution_df["% Isolated Contribution"] = (contribution_df["Isolated Reduction"] / baseline_kl) * 100
    else:
        contribution_df["% Contribution"] = 0.0
        contribution_df["% Isolated Contribution"] = 0.0

    # Also express each step relative to the TOTAL KL
    # reduction achieved by the full circuit (these sum to 100%), which is the
    # quantity the manuscript text refers to. "% Contribution" above is relative
    # to the baseline KL and does NOT sum to 100%.
    total_reduction = baseline_kl - contribution_df["KL Divergence"].iloc[-1]
    if total_reduction > 1e-12:
        contribution_df["% of Total Reduction"] = (-contribution_df["KL Change"] / total_reduction) * 100
    else:
        contribution_df["% of Total Reduction"] = 0.0

    contribution_df["CRX Angle (deg)"] = np.rad2deg(contribution_df["CRX Angle (rad)"])

    # --- 4. Generate Hub and Gene Set Summaries ---
    hubs = defaultdict(list)
    all_involved_genes = set()
    for _, row in contribution_df.iloc[1:].iterrows(): # Skip the baseline row
        interaction_parts = row["Interaction"].split(" -> ")
        control_gene, target_gene = interaction_parts[0], interaction_parts[1]
        hubs[control_gene].append(target_gene)
        all_involved_genes.add(control_gene)
        all_involved_genes.add(target_gene)

    # --- 5. Print All Summaries for Immediate Inspection ---
    print("\n--- Gate Contribution Analysis (Ablation Study) ---")
    df_display = contribution_df.copy()
    # Format columns for better printing
    formatters = {
        "KL Divergence": '{:.4f}',
        "KL Change": '{:.4f}',
        "Cumulative KL Reduction": '{:.4f}',
        "Isolated KL": '{:.4f}',
        "Isolated Reduction": '{:.4f}',
        "CRX Angle (rad)": '{:.4f}',
        "CRX Angle (deg)": '{:.2f}°',
        "% Contribution": '{:.2f}%',
        "% Isolated Contribution": '{:.2f}%',
        "% of Total Reduction": '{:.2f}%'
    }
    for col, f_str in formatters.items():
        if col in df_display.columns:
            df_display[col] = df_display[col].map(f_str.format)
    print(df_display.to_string())
    
    print("\n--- Hub-based Interaction Summary ---")
    for hub_gene, targets in sorted(hubs.items()):
        print(f"Gene '{hub_gene}' interacts with: {', '.join(sorted(targets))}")
        
    print("\n--- Full Set of Interacting Genes ---")
    print(sorted(list(all_involved_genes)))
    print("="*70)

    # --- 6. Return the Structured Data ---
    return contribution_df, dict(hubs), all_involved_genes


def plot_quantum_relay_network(
    topology,
    gene_list,
    gl_ct1,
    gl_ct2,
    valid_lr_pairs,
    contribution_df=None,
    title="Biologically Constrained Interaction Network",
    ct1_name="Cancer Cells",
    ct2_name="Fibroblasts",
    figsize=(10, 6),
    fname=None,
):
    """Interaction network, rendered via qxct.plotting.interaction_network so it
    matches every other figure (two columns, double-headed arrows, width = the
    gate's KL contribution). Signature unchanged.

    ``contribution_df`` (from analyze_and_summarize_network) sets the arrow
    widths via its ``KL Change`` column; ``valid_lr_pairs`` is accepted for
    backward compatibility but no longer needed for layout.
    """
    import matplotlib.pyplot as plt
    from .plotting import interaction_network
    n1 = len(gl_ct1)
    idx = {g: i for i, g in enumerate(gene_list)}
    contribs = None
    if contribution_df is not None:
        w = {}
        for _, row in contribution_df.iterrows():
            if "Baseline" in str(row.get("Interaction", "")):
                continue
            parts = str(row.get("Interaction", "")).split(" -> ")
            if len(parts) == 2 and "KL Change" in row:
                w[(parts[0].strip(), parts[1].strip())] = abs(float(row["KL Change"]))
        contribs = [w.get((gene_list[c], gene_list[t]), 0.0) for c, t in topology]
    fig, ax = plt.subplots(figsize=figsize)
    interaction_network(ax, list(topology), gene_list, n1, contributions=contribs,
                        name1=ct1_name, name2=ct2_name, bidirectional=True)
    ax.set_title(title, fontsize=12, pad=22)
    fig.tight_layout()
    if fname:
        fig.savefig(fname, bbox_inches="tight", transparent=True, dpi=300)
        print(f"Figure saved as {fname}")
    plt.show()


# ==============================================================================
# --- RESULTS I/O ---
# ==============================================================================
import json
import os

def save_results(
    path: str,
    cnot_sequence: list,
    crx_angles: list,
    gene_list: list = None,
    contribution_df=None,
    metadata: dict = None
):
    """
    Saves the full QuantumXCT result bundle to disk.

    Writes a JSON file containing the CNOT topology and optimized CRX angles,
    plus an optional CSV of the ablation study DataFrame.  Both files share the
    same base name so they stay together.

    File layout (example with path="results/ovarian_run1"):
        results/ovarian_run1.json   ← topology, angles, gene list, metadata
        results/ovarian_run1.csv    ← ablation DataFrame (if contribution_df given)

    Args:
        path (str): Output path *without* extension (e.g. "results/run1").
        cnot_sequence (list): Ordered list of (control_qubit, target_qubit) tuples.
        crx_angles (list): Optimized CRX rotation angles matching cnot_sequence order.
        gene_list (list, optional): Master gene list ordered by qubit index.
        contribution_df (pd.DataFrame, optional): Ablation study output from
            analyze_and_summarize_network().
        metadata (dict, optional): Any extra key-value pairs to store
            (e.g. nshots, threshold, dataset name).

    Returns:
        str: The base path used (without extension).
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    bundle = {
        "cnot_sequence": [list(gate) for gate in cnot_sequence],
        "crx_angles": list(crx_angles),
        "gene_list": gene_list,
        "metadata": metadata or {},
    }

    json_path = path + ".json"
    with open(json_path, "w") as f:
        json.dump(bundle, f, indent=2)
    print(f"Topology saved → {json_path}")

    if contribution_df is not None:
        csv_path = path + "_ablation.csv"
        contribution_df.to_csv(csv_path, index=False)
        print(f"Ablation results saved → {csv_path}")

    return path


def load_results(path: str):
    """
    Loads a QuantumXCT result bundle previously saved with save_results().

    Args:
        path (str): Base path without extension (e.g. "results/run1"),
                    or with ".json" extension — both are accepted.

    Returns:
        dict with keys:
            - cnot_sequence (list[tuple]): The CNOT topology as (control, target) tuples.
            - crx_angles (list[float]): Optimized CRX rotation angles.
            - gene_list (list | None): Gene names ordered by qubit index.
            - metadata (dict): Any extra metadata stored at save time.
    """
    json_path = path if path.endswith(".json") else path + ".json"

    with open(json_path, "r") as f:
        bundle = json.load(f)

    bundle["cnot_sequence"] = [tuple(gate) for gate in bundle["cnot_sequence"]]
    print(f"Results loaded from {json_path}")
    return bundle
