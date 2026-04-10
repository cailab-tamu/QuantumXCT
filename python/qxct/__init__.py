"""
qxct — QuantumXCT Python Package
==================================
Hybrid quantum-classical framework for inferring cell-cell communication
from single-cell RNA-seq data.

Modules
-------
encoding    Quantum state encoding (histograms → amplitude vectors)
circuits    Quantum circuit construction, topology search, and optimization
"""

from .encoding import (
    create_joint_histogram,
    create_percent_joint_histogram,
    count_boolean_vector_occurrences,
    plot_joint_histogram,
    normalize_counts,
)

from .circuits import (
    # State preparation
    build_initial_state_circuit,
    vector_normalize_dictionary_values,

    # Circuit construction
    concatenate_circuits_with_separate_measurements,
    apply_entanglement_topology,
    add_crx_gates_and_measurements_to_circuit,

    # Scoring
    get_probability_distribution,
    calculate_kl_divergence,
    score_circuit_kl_divergences,

    # Topology search — Algorithm 1 (N-wise iterative)
    find_cnot_candidates_from_state_diff,
    find_best_cnot_sequence_iterative_n_wise,

    # Topology search — Algorithm 2 (Multi-epoch greedy)
    find_best_cnot_sequence_multi_epoch,

    # Topology search — Algorithm 3 (QUBO + VQE hybrid)
    build_kl_divergence_matrix_interaction,
    kl_to_qubo_matrix,
    vqe_solver,
    evaluate_and_plot_ansatz,
    run_vqe_hybrid_search,

    # Angle optimization
    optimize_crx_angles,

    # Visualization & analysis
    plot_measurement_histograms,
    analyze_and_summarize_network,
    plot_quantum_relay_network,

    # Results I/O
    save_results,
    load_results,
)

__version__ = "0.1.0"
__author__ = "Selim Romero"
