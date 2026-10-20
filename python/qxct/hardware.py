"""
qxct.hardware
=============
Backend selection, error mitigation, and the post-optimization "assert quality"
run on a real quantum computer (or a noisy simulator).

The QuantumXCT search/optimization is normally done on a noiseless simulator.
Once the best CNOT topology and CRX angles are known, you usually want to run
that single, final circuit *once* on real hardware (with the error-mitigation
options you choose) to confirm the result survives device noise. That is what
:func:`assert_quality` does: it rebuilds the optimized circuit, executes it with
the requested mitigation, and reports the KL divergence against the target
distributions so you can compare hardware vs. the ideal simulation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2 as Sampler

from .circuits import (
    concatenate_circuits_with_separate_measurements,
    add_crx_gates_and_measurements_to_circuit,
    get_probability_distribution,
    calculate_kl_divergence,
    _process_target_state_input,
)
from .config import BackendConfig, MitigationConfig
# Scope helpers now live in qxct.scopes (they are general-purpose, not backend-
# specific). Re-exported here for backward compatibility.
from .scopes import (
    scope_of_cnot,
    filter_candidates_by_scope,
    enumerate_cnots,
    split_candidates_by_cell,
)


# ----------------------------------------------------------------------------
# Backend + sampler configuration
# ----------------------------------------------------------------------------
def resolve_backend(backend: Any = None, seed: Optional[int] = None) -> Any:
    """Return a usable backend; default to a local, seeded AerSimulator."""
    if backend is not None:
        return backend
    from .circuits import _GLOBAL_SEED
    return AerSimulator(seed_simulator=_GLOBAL_SEED if seed is None else int(seed))


def configure_sampler(backend: Any, mitigation: Optional[MitigationConfig] = None) -> Sampler:
    """Build a SamplerV2 on ``backend`` with the requested mitigation options.

    Options are set defensively: attributes that a given runtime/simulator
    version does not expose are skipped rather than raising, so the same code
    path works on a local AerSimulator and on real IBM hardware.
    """
    sampler = Sampler(mode=backend)
    if mitigation is None:
        return sampler

    opts = sampler.options

    def _try_set(root, attr, value):
        try:
            obj = getattr(root, attr)
        except Exception:
            return False
        try:
            setattr(root, attr, value)
            return True
        except Exception:
            return False

    # Dynamical decoupling
    try:
        opts.dynamical_decoupling.enable = bool(mitigation.dynamical_decoupling)
        if mitigation.dynamical_decoupling and mitigation.dd_sequence:
            _try_set(opts.dynamical_decoupling, "sequence_type", mitigation.dd_sequence)
    except Exception:
        pass

    # Gate / measurement twirling
    try:
        opts.twirling.enable_gates = bool(mitigation.twirling_gates)
        opts.twirling.enable_measure = bool(mitigation.twirling_measure)
    except Exception:
        pass

    # Readout-error mitigation.
    # NOTE: SamplerV2 has NO ``resilience`` / ``resilience_level``
    # options (those belong to EstimatorV2). The previous code tried
    # ``opts.resilience.measure_mitigation`` which always raised and silently fell
    # back to measurement twirling, and ``resilience_level`` was a silent no-op.
    # Measurement twirling only symmetrizes readout noise; it does not correct it.
    # Actual readout correction is now done in ``assert_quality`` with a tensored
    # (per-qubit) calibration, see ``_readout_calibration``.
    if mitigation.measurement_mitigation:
        _try_set(opts.twirling, "enable_measure", True)
    if mitigation.resilience_level is not None:
        import warnings
        warnings.warn("resilience_level is not a SamplerV2 option and is ignored; "
                      "readout correction is applied in assert_quality instead.")

    return sampler


# ----------------------------------------------------------------------------
# Build the final optimized circuit
# ----------------------------------------------------------------------------
def build_optimized_circuit(
    circ1: QuantumCircuit,
    circ2: QuantumCircuit,
    topology: List[Tuple[int, int]],
    angles: List[float],
) -> QuantumCircuit:
    """Rebuild the final circuit: the two initial-state circuits + the optimized
    CRX topology + measurements."""
    base = concatenate_circuits_with_separate_measurements(circ1, circ2)
    return add_crx_gates_and_measurements_to_circuit(base, circ1.num_qubits, topology, angles)


def _initialize_to_stateprep(circuit: QuantumCircuit) -> QuantumCircuit:
    """Replace ``Initialize`` (which inserts a reset on every qubit) by the
    equivalent ``StatePreparation`` (no resets). On hardware the qubits start in
    |0>, so the resets only add error and duration."""
    from qiskit.circuit.library import StatePreparation
    out = circuit.copy_empty_like()
    for inst in circuit.data:
        op = inst.operation
        if op.name == "initialize":
            op = StatePreparation(op.params)
        out.append(op, inst.qubits, inst.clbits)
    return out


def _readout_calibration(backend, qubits, shots, sampler_factory):
    """Tensored readout calibration on the *physical* qubits used by the circuit.
    Returns a list of 2x2 assignment matrices A_q with A_q[m, p] = P(measure m | prepared p)."""
    from qiskit import QuantumCircuit as _QC
    n = len(qubits)
    mats = []
    circs = []
    for prep in (0, 1):
        qc = _QC(max(qubits) + 1, n)
        if prep:
            for q in qubits:
                qc.x(q)
        qc.measure(list(qubits), list(range(n)))
        circs.append(qc)
    pm = generate_preset_pass_manager(backend=backend, optimization_level=0,
                                      initial_layout=list(range(max(qubits) + 1)))
    isa = [pm.run(c) for c in circs]
    res = sampler_factory().run(isa, shots=shots).result()
    counts = [r.data.c.get_counts() if hasattr(r.data, "c") else list(r.data.values())[0].get_counts()
              for r in res]
    for j in range(n):
        A = np.zeros((2, 2))
        for prep in (0, 1):
            tot = sum(counts[prep].values())
            ones = sum(v for k, v in counts[prep].items() if k[::-1][j] == "1")
            A[1, prep] = ones / tot
            A[0, prep] = 1 - ones / tot
        mats.append(A)
    return mats


def _apply_tensored_correction(counts: dict, mats, nbits: int) -> dict:
    """Invert tensored assignment matrices and project to the probability simplex."""
    p = np.zeros(2 ** nbits)
    tot = sum(counts.values())
    for k, v in counts.items():
        p[int(k, 2)] = v / tot
    t = p.reshape([2] * nbits)
    for j, A in enumerate(mats):             # bit j = qubit j = axis nbits-1-j
        Ainv = np.linalg.pinv(A)
        ax = nbits - 1 - j
        t = np.moveaxis(np.tensordot(Ainv, np.moveaxis(t, ax, 0), axes=(1, 0)), 0, ax)
    q = np.clip(t.reshape(-1), 0, None)
    q = q / q.sum() if q.sum() > 0 else q
    return {format(i, f"0{nbits}b"): float(q[i]) for i in range(2 ** nbits)}


# ----------------------------------------------------------------------------
# The hardware "assert quality" run
# ----------------------------------------------------------------------------
def assert_quality(
    circ1: QuantumCircuit,
    circ2: QuantumCircuit,
    topology: List[Tuple[int, int]],
    angles: List[float],
    state_vec_probs_target1,
    state_vec_probs_target2,
    backend_cfg: Optional[BackendConfig] = None,
    mitigation: Optional[MitigationConfig] = None,
    norm: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the optimized circuit once on a backend and score it against the target.

    Returns a dict with the measured counts, the KL divergence per register and
    the total, plus the backend name and the mitigation options actually used.
    Intended to be called with a real IBM backend (via ``backend_cfg.backend``)
    to confirm the simulator-optimized result survives device noise.
    """
    backend_cfg = backend_cfg or BackendConfig()
    mitigation = mitigation or MitigationConfig()
    backend = resolve_backend(backend_cfg.backend)
    shots = mitigation.shots or backend_cfg.shots

    circuit = build_optimized_circuit(circ1, circ2, topology, angles)
    circuit = _initialize_to_stateprep(circuit)

    # Transpile for the target backend.
    from .circuits import _GLOBAL_SEED
    pm = generate_preset_pass_manager(
        backend=backend, optimization_level=mitigation.optimization_level,
        seed_transpiler=_GLOBAL_SEED,          # reproducible layout/routing
    )
    isa_circuit = pm.run(circuit)
    ops = dict(isa_circuit.count_ops())
    n2q = sum(v for k, v in ops.items() if k in ("cx", "ecr", "cz", "rzz"))

    sampler = configure_sampler(backend, mitigation)
    job = sampler.run([isa_circuit], shots=shots)
    result = job.result()[0]

    counts1 = result.data.c_measure1.get_counts()
    counts2 = result.data.c_measure2.get_counts()

    n1 = circuit.cregs[0].size
    n2 = circuit.cregs[1].size
    sim1 = get_probability_distribution(counts1, n1, shots)
    sim2 = get_probability_distribution(counts2, n2, shots)

    t1, nq1 = _process_target_state_input(state_vec_probs_target1, norm)
    t2, nq2 = _process_target_state_input(state_vec_probs_target2, norm)
    target1 = {bin(i)[2:].zfill(nq1): float(np.abs(t1[i]) ** 2) for i in range(2 ** nq1)}
    target2 = {bin(i)[2:].zfill(nq2): float(np.abs(t2[i]) ** 2) for i in range(2 ** nq2)}

    kl1 = calculate_kl_divergence(sim1, target1)
    kl2 = calculate_kl_divergence(sim2, target2)

    # Exact (noiseless) reference for the same circuit, for a hardware-vs-ideal comparison
    from .exact import exact_score_circuit
    kl1_exact, kl2_exact = exact_score_circuit(
        build_optimized_circuit(circ1, circ2, topology, angles),
        state_vec_probs_target1, state_vec_probs_target2, norm=norm)

    # Optional readout-error correction (tensored calibration on the physical qubits)
    corrected = None
    if mitigation.measurement_mitigation:
        try:
            layout = isa_circuit.layout.final_index_layout()
            mats = _readout_calibration(backend, layout, shots,
                                        lambda: configure_sampler(backend, None))
            c1 = _apply_tensored_correction(counts1, mats[:n1], n1)
            c2 = _apply_tensored_correction(counts2, mats[n1:n1 + n2], n2)
            corrected = {"kl_measure1": calculate_kl_divergence(c1, target1),
                         "kl_measure2": calculate_kl_divergence(c2, target2)}
            corrected["kl_total"] = corrected["kl_measure1"] + corrected["kl_measure2"]
            corrected["assignment_matrices"] = [m.tolist() for m in mats]
        except Exception as e:                      # never lose the raw result
            corrected = {"error": repr(e)}

    backend_name = getattr(backend, "name", None)
    if callable(backend_name):
        backend_name = backend_name()

    return {
        "backend": backend_name or str(backend),
        "shots": shots,
        "n_gates": len(topology),
        "counts_measure1": counts1,
        "counts_measure2": counts2,
        "kl_measure1": kl1,
        "kl_measure2": kl2,
        "kl_total": kl1 + kl2,
        "kl_exact_reference": kl1_exact + kl2_exact,
        "kl_readout_corrected": corrected,
        "transpiled_depth": isa_circuit.depth(),
        "transpiled_2q_gates": n2q,
        "transpiled_ops": ops,
        "physical_layout": list(isa_circuit.layout.final_index_layout()) if isa_circuit.layout else None,
        "mitigation": {
            "dynamical_decoupling": mitigation.dynamical_decoupling,
            "twirling_gates": mitigation.twirling_gates,
            "twirling_measure": mitigation.twirling_measure,
            "measurement_mitigation": mitigation.measurement_mitigation,
            "resilience_level": mitigation.resilience_level,
        },
    }
