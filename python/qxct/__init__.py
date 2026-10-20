"""
qxct — QuantumXCT Python Package
==================================
Hybrid quantum-classical framework for inferring cell-cell communication
from single-cell RNA-seq data.

Modules
-------
encoding    Quantum state encoding (histograms → amplitude vectors)
circuits    Quantum circuit construction, topology search, and optimization
config      Configuration dataclasses (encoding, backend, search, mitigation)
hardware    Backend selection, error mitigation, hardware "assert quality" run
model       QuantumXct — the object-oriented orchestrator
exact       Exact (shot-noise-free) scoring, exact contribution tables, local-RY null
budget      Effective-degrees-of-freedom budget-matched baseline benchmark (v2)

Two ways to use qxct
--------------------
1. Object-oriented (recommended):
       from qxct import QuantumXct, RunConfig, EncodingConfig, SearchConfig
       qx = QuantumXct(RunConfig(encoding=EncodingConfig(norm="l1")), gene_list=genes)
       qx.encode(ct1_basal, ct2_basal, ct1_inter, ct2_inter)
       result = qx.run()

2. Functional (the original API, still fully supported).
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
    find_cnot_candidates_dense,
    find_cnot_candidates_sparse,
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
    compare_angle_initializations,

    # Visualization & analysis
    plot_measurement_histograms,
    analyze_and_summarize_network,
    plot_quantum_relay_network,

    # Results I/O
    save_results,
    load_results,

    # Reproducibility
    set_global_seed,
)

# --- Object-oriented API ---
from .config import (
    RunConfig,
    EncodingConfig,
    BackendConfig,
    SearchConfig,
    OptimizerConfig,
    MitigationConfig,
)
from .model import QuantumXct, QuantumXctResult
from .analysis import (
    compare_scopes,
    run_scope,
    cumulative_ablation_table,
    isolated_contribution_table,
    baseline_kl,
    plot_scope_histograms,
    post_optimization_distributions,
    plot_scope_report,
    compare_norms,
)
from .baselines import (
    BENCHMARK_B_CAPTION,
    compare_baselines,
    benchmark_budget_matched,
    ot_sinkhorn_baseline,
    tn_transform_baseline,
    born_machine_baseline,
    quantumxct_baseline,
)
from .hardware import (
    assert_quality,
    build_optimized_circuit,
    configure_sampler,
    resolve_backend,
)
from .scopes import (
    scope_of_cnot,
    filter_candidates_by_scope,
    split_candidates_by_cell,
    enumerate_cnots,
)
from .resources import (
    load_lr_resource,
    curate_lr_pairs,
)
from .metrics import (
    evaluate_edge_recovery,
    synthetic_ground_truth,
    SIMULATOR_EDGES,
)
# Exact scoring and effective-degrees-of-freedom budget benchmark
from .exact import (
    ExactScorer,
    exact_contribution_table,
    exact_score_circuit,
    use_exact_scoring,
)
from .budget import benchmark_budget_matched_v2, mps_dof, mpo_dof

__version__ = "0.2.0"
__author__ = "Selim Romero"
