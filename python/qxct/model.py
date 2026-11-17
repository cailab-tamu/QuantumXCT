"""
qxct.model
==========
The object-oriented front door to QuantumXCT.

``QuantumXct`` bundles a :class:`~qxct.config.RunConfig` with the state it builds
up as it runs, and exposes the pipeline as a handful of methods instead of a long
chain of free functions each taking a dozen keyword arguments:

    from qxct import QuantumXct, RunConfig, EncodingConfig, SearchConfig
    qx = QuantumXct(RunConfig(
        encoding=EncodingConfig(norm="l1"),
        search=SearchConfig(candidate_scope="all"),
    ), gene_list=genes)

    qx.encode(counts_ct1_mono, counts_ct2_mono, counts_ct1_co, counts_ct2_co)
    result = qx.run()                      # screen -> search -> optimize -> analyze
    hw = qx.assert_on_hardware(backend=ibm_backend)   # validate on real device

The heavy lifting still lives in :mod:`qxct.circuits`; this class just wires those
functions together and threads the configuration through consistently (in
particular the L1/L2 encoding, which is applied once, here, to every state).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit

from .circuits import (
    vector_normalize_dictionary_values,
    build_initial_state_circuit,
    find_cnot_candidates_from_state_diff,
    find_best_cnot_sequence_multi_epoch,
    find_best_cnot_sequence_iterative_n_wise,
    optimize_crx_angles,
    analyze_and_summarize_network,
    save_results,
    set_global_seed,
)
from .config import RunConfig
from .hardware import assert_quality
from .scopes import (
    filter_candidates_by_scope,
    split_candidates_by_cell,
    enumerate_cnots,
)


@dataclass
class QuantumXctResult:
    """Container for the outputs of a completed run."""
    candidates: List[Tuple[int, int]] = field(default_factory=list)
    topology: List[Tuple[int, int]] = field(default_factory=list)
    angles: List[float] = field(default_factory=list)
    kl_sum: float = float("inf")
    contribution_df: Optional[pd.DataFrame] = None
    hubs: Optional[dict] = None
    involved_genes: Optional[set] = None
    angle_history: List[float] = field(default_factory=list)


class QuantumXct:
    """Config-driven orchestrator for the QuantumXCT pipeline."""

    def __init__(self, config: Optional[RunConfig] = None, gene_list: Optional[List[str]] = None):
        self.config = config or RunConfig()
        set_global_seed(self.config.seed)   # reproducibility: seed random / numpy / Aer sampling
        self.gene_list = gene_list

        # State populated by encode()
        self.circ1: Optional[QuantumCircuit] = None
        self.circ2: Optional[QuantumCircuit] = None
        self.initial1: Optional[dict] = None
        self.initial2: Optional[dict] = None
        self.target1: Optional[dict] = None
        self.target2: Optional[dict] = None
        self.ng_circ1: Optional[int] = None

        self.result = QuantumXctResult()

    # ------------------------------------------------------------------ encode
    def encode(
        self,
        counts_ct1_basal: dict,
        counts_ct2_basal: dict,
        counts_ct1_interacting: dict,
        counts_ct2_interacting: dict,
    ) -> "QuantumXct":
        """Turn the four histograms into amplitudes + initial-state circuits.

        The basal (e.g. mono-culture) distributions define the initial state; the
        interacting (e.g. co-culture) distributions define the target. All four
        are normalized once with the configured L1/L2 convention, so every
        downstream comparison is in the same encoding.
        """
        norm = self.config.encoding.norm
        self.initial1 = vector_normalize_dictionary_values(counts_ct1_basal, norm)
        self.initial2 = vector_normalize_dictionary_values(counts_ct2_basal, norm)
        self.target1 = vector_normalize_dictionary_values(counts_ct1_interacting, norm)
        self.target2 = vector_normalize_dictionary_values(counts_ct2_interacting, norm)

        # norm=None below: the dicts already hold amplitudes.
        self.circ1 = build_initial_state_circuit(self.initial1, norm=None)
        self.circ2 = build_initial_state_circuit(self.initial2, norm=None)
        self.ng_circ1 = self.circ1.num_qubits
        return self

    def set_states(self, circ1, circ2, initial1, initial2, target1, target2) -> "QuantumXct":
        """Alternative to encode(): provide pre-built circuits / amplitude dicts."""
        self.circ1, self.circ2 = circ1, circ2
        self.initial1, self.initial2 = initial1, initial2
        self.target1, self.target2 = target1, target2
        self.ng_circ1 = circ1.num_qubits
        return self

    def _require_encoded(self):
        if self.circ1 is None:
            raise RuntimeError("Call encode() (or set_states()) before running the pipeline.")

    # -------------------------------------------------------------- candidates
    def screen_candidates(self) -> List[Tuple[int, int]]:
        """Build the candidate CNOT set for the configured scope (GRN / LR).

        This is the single source of truth for candidate selection and applies to
        *any* optimization, not just the benchmark:

          - scope='all'   (GRN + LR) : density-matrix screening, no scope filter.
          - scope='intra' (GRN only) : density-matrix screening, then keep only
                                        within-cell links.
          - scope='inter' (LR only)  : ALL cross-register pairs, both directions,
                                        with NO density-matrix screening and NO
                                        GRN links.
        """
        self._require_encoded()
        s = self.config.search

        if s.candidate_scope == "inter" or not s.screen:
            # No density-matrix screening: enumerate every candidate pair of the
            # scope (richer search, memory grows with N(N-1); off by default).
            ng_total = self.circ1.num_qubits + self.circ2.num_qubits
            scoped = enumerate_cnots(ng_total, self.ng_circ1, scope=s.candidate_scope)
        else:
            candidates, _ = find_cnot_candidates_from_state_diff(
                self.initial1, self.initial2, self.target1, self.target2,
                threshold=s.threshold, search_mode=s.search_mode,
                verbose_print=True, norm=None,
            )
            scoped = filter_candidates_by_scope(candidates, self.ng_circ1, s.candidate_scope)

        self.result.candidates = scoped
        return scoped

    # ------------------------------------------------------------------ search
    def search_topology(self, candidates: Optional[List[Tuple[int, int]]] = None):
        """Run the configured topology search over (scoped) candidates."""
        self._require_encoded()
        s = self.config.search
        shots = self.config.backend.shots
        if candidates is None:
            candidates = self.result.candidates or self.screen_candidates()

        if s.algorithm == "multi_epoch":
            topo, kl = find_best_cnot_sequence_multi_epoch(
                self.circ1, self.circ2, self.initial1, self.initial2,
                self.target1, self.target2,
                cnot_search_candidates=candidates,
                n_epochs=s.n_epochs, min_cnot_depth=s.min_cnot_depth,
                nshots=shots, threshold=s.threshold, search_mode=s.search_mode,
                kl_tol=s.kl_tol,
                max_greedy_depth=s.max_greedy_depth,
            )
        elif s.algorithm == "iterative_n_wise":
            _cands, topo, kl = find_best_cnot_sequence_iterative_n_wise(
                self.circ1, self.circ2, self.initial1, self.initial2,
                self.target1, self.target2,
                nshots=shots, threshold=s.threshold, search_mode=s.search_mode,
                nchoose=s.nchoose, kl_tol=s.kl_tol,
                cnot_search_candidates=candidates,
            )
        elif s.algorithm == "vqe_hybrid":
            return self._search_vqe_hybrid(candidates)
        else:
            raise ValueError(
                f"Unknown search algorithm {s.algorithm!r}. Use 'multi_epoch', "
                f"'iterative_n_wise' or 'vqe_hybrid'."
            )

        self.result.topology = topo
        self.result.kl_sum = kl
        return topo, kl

    def _search_vqe_hybrid(self, candidates=None):
        """QUBO + VQE hybrid search (algorithm='vqe_hybrid').

        Mirrors the notebook flow: build the KL-divergence interaction matrix,
        convert it to a QUBO / Ising Hamiltonian, minimize with a VQE over an
        EfficientSU2 ansatz, then run the classical hybrid refinement over the
        top-k VQE solutions.

        The ansatz is transpiled for the backend at
        ``config.search.vqe_optimization_level``. Default None = automatic: level 0
        on an ideal simulator, level 3 on a real device or device emulation, where
        Qiskit must map it onto the qubit connectivity and native gates.
        """
        from qiskit_optimization import QuadraticProgram
        from qiskit.circuit.library import EfficientSU2
        from qiskit.compiler import transpile

        from .circuits import (
            build_kl_divergence_matrix_interaction, kl_to_qubo_matrix,
            vqe_solver, evaluate_and_plot_ansatz, run_vqe_hybrid_search,
        )
        from .hardware import resolve_backend

        s = self.config.search
        shots = self.config.backend.shots
        backend = resolve_backend(self.config.backend.backend)

        kl_matrix, _cnot_to_index, initial_cnot_config, initial_kl_sum = \
            build_kl_divergence_matrix_interaction(
                self.circ1, self.circ2, self.initial1, self.initial2,
                self.target1, self.target2, nshots=shots,
                threshold=s.threshold, search_mode=s.search_mode,
                include_single_cnot_kl=True, cnot_search_candidates=candidates,
            )
        if not initial_cnot_config:
            self.result.topology, self.result.kl_sum = [], initial_kl_sum
            return [], initial_kl_sum

        qubo = kl_to_qubo_matrix(kl_matrix, initial_kl_sum)
        n = qubo.shape[0]
        qp = QuadraticProgram(name="CnotOptimization")
        for i in range(n):
            qp.binary_var(name=f"x{i}")
        qp.minimize(quadratic=qubo)
        hamiltonian, _offset = qp.to_ising()

        ansatz = EfficientSU2(num_qubits=n, entanglement=s.vqe_entanglement, reps=s.vqe_reps)
        expanded = transpile(ansatz, basis_gates=["u", "cx"], optimization_level=0)

        _res, opt_params, _cost = vqe_solver(
            expanded, hamiltonian, backend,
            optimizer_method=self.config.optimizer.method, niter=s.vqe_niter,
            optimization_level=s.vqe_optimization_level,
        )
        counts, _bound = evaluate_and_plot_ansatz(
            ansatz=expanded, params=opt_params, backend=backend, shots=1024, show_plot=False,
            optimization_level=s.vqe_optimization_level,
        )
        topo, kl, _rank = run_vqe_hybrid_search(
            vqe_counts=counts, initial_cnot_config=initial_cnot_config,
            circ1=self.circ1, circ2=self.circ2,
            state_probs_initial1=self.initial1, state_probs_initial2=self.initial2,
            state_vec_probs_target1=self.target1, state_vec_probs_target2=self.target2,
            num_solutions_to_test=s.vqe_top_k, nshots=shots, threshold=s.threshold,
            search_mode=s.search_mode, kl_tol=s.kl_tol,
            max_greedy_depth=s.max_greedy_depth,
        )
        topo = topo or []
        self.result.topology, self.result.kl_sum = topo, kl
        return topo, kl

    def search_topology_grn_sequential(self):
        """GRN-only search, one cell TYPE (qubit register) at a time.

        "Cell" here means a cell type, i.e. one of the two qubit registers
        (register 1 = cell type 1, qubits 0..n1-1; register 2 = cell type 2,
        qubits n1..N-1), never an individual cell. The candidates are restricted to
        within-register (intra, GRN) links, split by register with
        ``split_candidates_by_cell``; register 1's internal topology is searched to
        convergence, then register 2's, and the two are concatenated. Cross-register
        (LR) links are excluded, so this is the "within-cell only" baseline.
        """
        self._require_encoded()
        s = self.config.search
        shots = self.config.backend.shots
        candidates = self.result.candidates or self.screen_candidates()
        groups = split_candidates_by_cell(candidates, self.ng_circ1)

        combined: List[Tuple[int, int]] = []
        for cell_key in ("cell1", "cell2"):
            cell_cands = groups[cell_key]
            if not cell_cands:
                continue
            topo, _ = find_best_cnot_sequence_multi_epoch(
                self.circ1, self.circ2, self.initial1, self.initial2,
                self.target1, self.target2,
                cnot_search_candidates=cell_cands,
                n_epochs=s.n_epochs,   # None => max (capped to this cell's candidate count inside)
                min_cnot_depth=s.min_cnot_depth, nshots=shots,
                threshold=s.threshold, search_mode=s.search_mode,
                kl_tol=s.kl_tol,
                max_greedy_depth=s.max_greedy_depth,
            )
            combined.extend(topo)

        self.result.topology = combined
        return combined

    # ---------------------------------------------------------------- optimize
    def optimize_angles(self, topology: Optional[List[Tuple[int, int]]] = None):
        """Optimize CRX angles for a fixed topology."""
        self._require_encoded()
        topology = topology if topology is not None else self.result.topology
        o = self.config.optimizer
        angles, kl, _hist = optimize_crx_angles(
            self.circ1, self.circ2, self.target1, self.target2,
            cnot_topology=topology, nshots=self.config.backend.shots,
            etol=o.etol, opt_method=o.method, initial_angle_value=o.initial_angle,
        )
        self.result.angles = angles
        self.result.kl_sum = kl
        self.result.angle_history = list(_hist) if _hist else []
        return angles, kl

    # ----------------------------------------------------------------- analyze
    def analyze(self, nshots: Optional[int] = None):
        """Ablation study + hub summary for the current topology & angles."""
        self._require_encoded()
        if self.gene_list is None:
            raise RuntimeError("analyze() needs gene_list (pass it to QuantumXct(...)).")
        df, hubs, genes = analyze_and_summarize_network(
            self.circ1, self.circ2, self.result.topology, self.result.angles,
            self.target1, self.target2, self.gene_list,
            nshots=nshots or self.config.backend.shots,
        )
        self.result.contribution_df = df
        self.result.hubs = hubs
        self.result.involved_genes = genes
        return df

    # ----------------------------------------------------------------- plot
    def plot_report(self, gl_ct1, gl_ct2, valid_lr_pairs, out_dir=".",
                    ct1_name="Cell type 1", ct2_name="Cell type 2",
                    nshots: Optional[int] = None, show: bool = True):
        """Print the full figure set for the completed run: optimized circuit
        diagram, measurement histograms, 3-way (mono/QuantumXCT/co) occupancy,
        and the interaction network colored by cell type. Saves to ``out_dir``.
        """
        self._require_encoded()
        from .analysis import plot_scope_report
        res = {
            "scope": self.config.search.candidate_scope,
            "topology": self.result.topology, "angles": self.result.angles,
            "final_kl": self.result.kl_sum,
            "circ1": self.circ1, "circ2": self.circ2,
            "target1": self.target1, "target2": self.target2,
            "basal1": self.initial1, "basal2": self.initial2,
            "nshots": nshots or self.config.backend.shots,
            "angle_history": self.result.angle_history,
        }
        return plot_scope_report(
            res, gl_ct1=gl_ct1, gl_ct2=gl_ct2, valid_lr_pairs=valid_lr_pairs,
            out_dir=out_dir, scope=self.config.search.candidate_scope,
            nshots=res["nshots"], ct1_name=ct1_name, ct2_name=ct2_name, show=show,
        )

    # ------------------------------------------------------------- hardware
    def assert_on_hardware(self, backend: Any = None, mitigation=None) -> Dict[str, Any]:
        """Validate the optimized circuit on a real backend with error mitigation."""
        self._require_encoded()
        bcfg = self.config.backend
        if backend is not None:
            import dataclasses
            bcfg = dataclasses.replace(bcfg, backend=backend)
        return assert_quality(
            self.circ1, self.circ2, self.result.topology, self.result.angles,
            self.target1, self.target2,
            backend_cfg=bcfg, mitigation=mitigation or self.config.mitigation, norm=None,
        )

    # ---------------------------------------------------------------- convenience
    def run(self, analyze: bool = True) -> QuantumXctResult:
        """Screen -> search -> optimize (-> analyze). Returns the result bundle."""
        self.screen_candidates()
        if self.config.search.candidate_scope == "intra":
            self.search_topology_grn_sequential()
        else:
            self.search_topology()
        self.optimize_angles()
        if analyze and self.gene_list is not None:
            self.analyze()
        return self.result

    def save(self, path: str, metadata: Optional[dict] = None) -> str:
        """Persist topology, angles, gene list and ablation table via save_results."""
        meta = {
            "norm": self.config.encoding.norm,
            "candidate_scope": self.config.search.candidate_scope,
            "shots": self.config.backend.shots,
            "kl_sum": self.result.kl_sum,
        }
        meta.update(metadata or {})
        return save_results(
            path, self.result.topology, self.result.angles,
            gene_list=self.gene_list, contribution_df=self.result.contribution_df,
            metadata=meta,
        )
