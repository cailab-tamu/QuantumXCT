"""
qxct.config
===========
Configuration objects for the QuantumXCT object-oriented API.

Historically every stage of the pipeline took its own long list of keyword
arguments (backend, shots, threshold, kl_tol, max_iters, error-mitigation flags,
...) scattered across half a dozen function calls. These dataclasses group those
knobs into a few cohesive, self-documenting objects so a user configures the run
in one place and passes them around as a bundle.

    from qxct.config import RunConfig, EncodingConfig, SearchConfig
    cfg = RunConfig(
        encoding=EncodingConfig(norm="l1"),            # L1 is the default
        search=SearchConfig(candidate_scope="all"),    # GRN + LR
    )

Every field has a sensible default, so ``RunConfig()`` alone is a valid,
ready-to-run configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np


# ----------------------------------------------------------------------------
# Encoding
# ----------------------------------------------------------------------------
@dataclass
class EncodingConfig:
    """How single-cell histograms are turned into quantum-state amplitudes.

    norm:
        'l1' (default) -> amplitude = sqrt(count / sum(count)); measurement
                          probability is proportional to the frequency
                          (frequency-preserving encoding).
        'l2'           -> amplitude = count / ||count||_2; measurement
                          probability is proportional to count^2 (original
                          paper encoding, emphasizes the most common states).

    The same convention is applied to the initial-state circuits AND to the
    target distributions and density-matrix screening, so the KL comparison is
    always between two states in the same encoding.
    """
    norm: str = "l1"

    def __post_init__(self):
        if self.norm not in ("l1", "l2"):
            raise ValueError(f"EncodingConfig.norm must be 'l1' or 'l2', got {self.norm!r}")


# ----------------------------------------------------------------------------
# Backend / execution
# ----------------------------------------------------------------------------
@dataclass
class BackendConfig:
    """Where and how circuits are executed during the search/optimization.

    backend:
        None (default) -> a local AerSimulator is created on demand.
        Otherwise any Qiskit BackendV2 (Aer or a real IBM backend / session).
    shots:
        Number of shots per circuit evaluation.
    optimization_level:
        Transpiler preset optimization level used while scoring circuits.
    """
    backend: Any = None
    shots: int = 1000
    optimization_level: int = 1


# ----------------------------------------------------------------------------
# Topology search
# ----------------------------------------------------------------------------
@dataclass
class SearchConfig:
    """Controls candidate screening and the CNOT/CRX topology search.

    algorithm:
        'multi_epoch'      -> find_best_cnot_sequence_multi_epoch (default)
        'iterative_n_wise' -> find_best_cnot_sequence_iterative_n_wise
        'vqe_hybrid'       -> QUBO + VQE hybrid pipeline
    candidate_scope:
        'all'   -> GRN + LR: every candidate CNOT (intra- and inter-register).
        'intra' -> GRN only: control and target within the SAME cell register.
        'inter' -> LR only: control and target in DIFFERENT cell registers.
    threshold / search_mode:
        Density-matrix-difference screening (see find_cnot_candidates_from_state_diff).
    kl_tol:
        Significance threshold; a longer sequence is only kept if it beats the
        shorter one by more than kl_tol ("Occam's razor").
    """
    algorithm: str = "multi_epoch"
    candidate_scope: str = "all"          # 'all' | 'intra' | 'inter'
    threshold: float = 0.1
    search_mode: str = "pos"              # 'pos' | 'neg' | 'abs'
    kl_tol: float = 0.05
    n_epochs: Optional[int] = None        # None (default) = MAX: one epoch per candidate (standardized, most thorough)
    min_cnot_depth: int = 1
    max_greedy_depth: int = 30            # a.k.a. "max iters"/depth for the greedy search
    screen: bool = True                   # False -> enumerate ALL candidate pairs (no density-matrix
                                          # screening): richer search, but memory grows with N(N-1);
                                          # implemented, not the default, for memory reasons.
    nchoose: int = 2                      # pair size for the n-wise search
    # VQE-hybrid (algorithm='vqe_hybrid') settings
    vqe_top_k: int = 3                    # #VQE solutions tested in the hybrid search
    vqe_reps: int = 1                     # EfficientSU2 repetitions
    vqe_entanglement: str = "circular"    # linear | circular | full | pairwise | ...
    vqe_niter: int = 200                  # VQE optimizer iterations
    vqe_optimization_level: Optional[int] = None  # Qiskit transpiler level (0-3) for the VQE ansatz.
                                          # None (default) = automatic: 0 on an ideal simulator
                                          # (no qubit-connectivity map), 3 on a real device or a
                                          # device emulation (e.g. AerSimulator.from_backend).
                                          # Set an int to force a level.

    def __post_init__(self):
        if self.candidate_scope not in ("all", "intra", "inter"):
            raise ValueError(
                f"SearchConfig.candidate_scope must be 'all', 'intra' or 'inter', "
                f"got {self.candidate_scope!r}"
            )
        if self.search_mode not in ("pos", "neg", "abs"):
            raise ValueError(f"SearchConfig.search_mode must be 'pos', 'neg' or 'abs'")


# ----------------------------------------------------------------------------
# CRX angle optimization
# ----------------------------------------------------------------------------
@dataclass
class OptimizerConfig:
    """Classical optimizer settings for CRX-angle refinement."""
    method: str = "COBYLA"                # e.g. "COBYLA", "L-BFGS-B"
    etol: float = 1e-6
    initial_angle: float = np.pi / 2      # start angle optimization at the topology-search value CRX(pi/2), then fine-tune (0 tends to get stuck at baseline)
    max_iters: int = 500


# ----------------------------------------------------------------------------
# Error mitigation (hardware "assert quality" run)
# ----------------------------------------------------------------------------
@dataclass
class MitigationConfig:
    """Error-mitigation toggles for a real-hardware validation run.

    Each option is independent so a user can enable only the subset they want.
    They map onto qiskit-ibm-runtime SamplerV2 options; on a local simulator the
    flags are harmless no-ops.
    """
    dynamical_decoupling: bool = False    # sampler.options.dynamical_decoupling.enable
    twirling_gates: bool = False          # sampler.options.twirling.enable_gates
    twirling_measure: bool = False        # sampler.options.twirling.enable_measure
    measurement_mitigation: bool = False  # readout/TREX measurement-error mitigation
    resilience_level: Optional[int] = None  # NOT a SamplerV2 option (ignored with a warning); kept for API compatibility
    dd_sequence: str = "XX"               # DD sequence type when dynamical_decoupling
    optimization_level: int = 3           # transpiler level for the hardware circuit
    shots: int = 4000                     # shots for the validation run (usually more)

    def any_enabled(self) -> bool:
        return bool(
            self.dynamical_decoupling
            or self.twirling_gates
            or self.twirling_measure
            or self.measurement_mitigation
            or (self.resilience_level is not None)
        )

    @classmethod
    def all_on(cls, resilience_level: Optional[int] = None, shots: int = 4000) -> "MitigationConfig":
        """Convenience constructor turning every mitigation option on."""
        return cls(
            dynamical_decoupling=True,
            twirling_gates=True,
            twirling_measure=True,
            measurement_mitigation=True,
            resilience_level=resilience_level,
            shots=shots,
        )


# ----------------------------------------------------------------------------
# Top-level bundle
# ----------------------------------------------------------------------------
@dataclass
class RunConfig:
    """One object holding every sub-configuration for a QuantumXCT run.

    seed: master random seed (default 42). It is applied to Python's `random`,
    NumPy, the Aer simulator (shot sampling), and the tensor-network baselines,
    so an entire run is reproducible. Change it to get an independent replicate.
    """
    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    mitigation: MitigationConfig = field(default_factory=MitigationConfig)
    seed: int = 42
