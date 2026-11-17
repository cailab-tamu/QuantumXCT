"""
Integrity tests for the core numerics.  Run:  python -m pytest -q tests
They cover L1 encoding (|amp|^2 = empirical frequency), bit ordering on
hardware-style transpilation, and the exact evaluator.
"""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qxct import (vector_normalize_dictionary_values as vn, build_initial_state_circuit,
                  score_circuit_kl_divergences)
from qxct.circuits import (concatenate_circuits_with_separate_measurements,
                           add_crx_gates_and_measurements_to_circuit,
                           find_cnot_candidates_from_state_diff,
                           find_cnot_candidates_dense, find_cnot_candidates_sparse)
from qxct.encoding import create_joint_histogram
from qxct.exact import ExactScorer, apply_crx, marginals, exact_score_circuit


def _synth(n, seed, shift, cells=400):
    rng = np.random.default_rng(seed)
    p = np.clip(rng.uniform(.25, .75, n) + shift, .05, .95)
    return dict(create_joint_histogram((rng.random((cells, n)) < p).astype(int))[0])


def test_l1_encoding_is_frequency_preserving():
    c = {"001": 30, "100": 50, "111": 20}
    a = vn(c, "l1")
    assert np.allclose([a[k] ** 2 for k in c], [0.3, 0.5, 0.2])
    a2 = vn(c, "l2")
    tot = sum(v * v for v in c.values())
    assert np.allclose([a2[k] ** 2 for k in c], [v * v / tot for v in c.values()])


def test_initial_state_circuit_reproduces_frequencies():
    c = _synth(3, 1, 0.0)
    qc = build_initial_state_circuit(vn(c, "l1"))
    p = Statevector(qc).probabilities_dict()
    tot = sum(c.values())
    for k, v in c.items():
        assert abs(p.get(k, 0) - v / tot) < 1e-10


def test_bit_order_survives_device_transpilation():
    """A gene pattern encoded as key 'x' must come back under key 'x' after
    transpiling to a 127-qubit device layout (noiseless run of the ISA circuit)."""
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime.fake_provider import FakeBrisbane
    from qiskit_ibm_runtime import SamplerV2
    from qiskit_aer import AerSimulator
    c1 = {"01": 70, "10": 30}
    c2 = {"001": 100}
    base = concatenate_circuits_with_separate_measurements(
        build_initial_state_circuit(vn(c1, "l1")), build_initial_state_circuit(vn(c2, "l1")))
    qc = add_crx_gates_and_measurements_to_circuit(base, 2, [], [])
    pm = generate_preset_pass_manager(backend=FakeBrisbane(), optimization_level=3)
    isa = pm.run(qc.decompose())
    res = SamplerV2(mode=AerSimulator(seed_simulator=1)).run([isa], shots=4000).result()[0]
    m1, m2 = res.data.c_measure1.get_counts(), res.data.c_measure2.get_counts()
    assert set(m1) <= {"01", "10"} and m1["01"] > m1["10"]
    assert m2 == {"001": 4000}


def test_exact_crx_matches_qiskit():
    rng = np.random.default_rng(0)
    v = rng.normal(size=32) + 1j * rng.normal(size=32); v /= np.linalg.norm(v)
    qc, psi = QuantumCircuit(5), v.copy()
    for c, t, th in [(0, 3, .7), (4, 1, 2.1), (2, 0, 1.3)]:
        qc.crx(th, c, t); psi = apply_crx(psi, 5, c, t, th)
    assert np.allclose(Statevector(v).evolve(qc).data, psi)


def test_exact_scorer_agrees_with_sampled_scorer():
    i1, i2, t1, t2 = _synth(2, 1, 0), _synth(3, 2, 0), _synth(2, 1, .2), _synth(3, 2, .2)
    a = [vn(x, "l1") for x in (i1, i2, t1, t2)]
    base = concatenate_circuits_with_separate_measurements(build_initial_state_circuit(a[0]),
                                                           build_initial_state_circuit(a[1]))
    topo, ang = [(0, 3), (4, 1)], [1.2, 0.8]
    circ = add_crx_gates_and_measurements_to_circuit(base, 2, topo, ang)
    ex = sum(exact_score_circuit(circ, a[2], a[3]))
    sc = ExactScorer(*a)
    assert abs(ex - sc.kl(topo, ang)) < 1e-9
    k1, k2 = score_circuit_kl_divergences(circ, a[2], a[3], nshots=200000)
    assert abs((k1 + k2) - ex) < 0.01


def test_single_inter_gate_acts_as_local_channel():
    """Proposition S1: one inter-register CRX leaves the control register's
    marginal unchanged and mixes the target register's marginal with weight
    P(control = 1)."""
    rng = np.random.default_rng(3)
    a = np.abs(rng.normal(size=8)); a /= np.linalg.norm(a)
    b = np.abs(rng.normal(size=8)); b /= np.linalg.norm(b)
    psi = np.kron(b, a); c, t, th = 1, 4, 1.1
    m1, m2 = marginals(apply_crx(psi, 6, c, t, th), 3, 3)
    pc1 = sum((a ** 2)[i] for i in range(8) if (i >> c) & 1)
    qc = QuantumCircuit(3); qc.rx(th, t - 3)
    assert np.allclose(m1, a ** 2)
    assert np.allclose(m2, (1 - pc1) * b ** 2 + pc1 * Statevector(b).evolve(qc).probabilities())


@pytest.mark.parametrize("mode", ["pos", "abs"])
def test_candidate_finders_are_equivalent(mode):
    i1, i2, t1, t2 = _synth(2, 1, 0), _synth(4, 2, 0), _synth(2, 1, .25), _synth(4, 2, .25)
    args = [vn(x, "l1") for x in (i1, i2, t1, t2)]
    ref, _ = find_cnot_candidates_from_state_diff(*args, threshold=0.005, search_mode=mode)
    d, _ = find_cnot_candidates_dense(*args, threshold=0.005, search_mode=mode, precision="float64")
    s, _ = find_cnot_candidates_sparse(*args, threshold=0.005, search_mode=mode, precision="float64")
    assert ref == d == s


def test_sampler_options_do_not_silently_claim_readout_mitigation():
    from qxct import MitigationConfig
    from qxct.hardware import configure_sampler
    from qiskit_aer import AerSimulator
    s = configure_sampler(AerSimulator(), MitigationConfig.all_on())
    assert s.options.dynamical_decoupling.enable is True
    assert s.options.twirling.enable_gates is True
    assert not hasattr(s.options, "resilience")     # SamplerV2 has no resilience options


# ---- added with the N-wise / VQE scope fix and the 10+10 scaling path (2026-09-28) ----
@pytest.mark.parametrize("mode", ["pos", "neg", "abs"])
@pytest.mark.parametrize("thr", [0.002, 0.02])
def test_pruned_sparse_screening_matches_dense(mode, thr):
    """Threshold pruning in find_cnot_candidates_sparse keeps exactly the dense candidates."""
    for seed in range(4):
        args = [vn(x, "l1") for x in (_synth(3, seed, 0), _synth(4, seed + 9, 0),
                                       _synth(3, seed, .2), _synth(4, seed + 9, .2))]
        ref, _ = find_cnot_candidates_from_state_diff(*args, threshold=thr, search_mode=mode)
        s, _ = find_cnot_candidates_sparse(*args, threshold=thr, search_mode=mode, precision="float64")
        assert sorted(ref) == sorted(s)


def test_fast_statevector_matches_qiskit():
    """exact._fast_statevector (state preparations applied directly) == Statevector(qc)."""
    from qxct.exact import _fast_statevector
    rng = np.random.default_rng(5)
    for _ in range(5):
        v1 = rng.random(8); v1 /= np.linalg.norm(v1); v2 = rng.random(4); v2 /= np.linalg.norm(v2)
        qc = QuantumCircuit(5); qc.initialize(v1, range(3)); qc.initialize(v2, range(3, 5))
        for _ in range(4):
            c, t = rng.choice(5, 2, replace=False); qc.crx(float(rng.random() * 3), int(c), int(t))
        assert np.allclose(Statevector(qc).data, _fast_statevector(qc).data, atol=1e-12)


@pytest.mark.parametrize("algorithm", ["iterative_n_wise", "multi_epoch"])
@pytest.mark.parametrize("scope", ["intra", "inter"])
def test_search_respects_candidate_scope(algorithm, scope):
    """N-wise (and VQE) used to re-screen internally and ignore GRN/LR scope."""
    from qxct import QuantumXct, RunConfig, SearchConfig, BackendConfig
    from qxct.exact import use_exact_scoring
    prev = use_exact_scoring(True)
    try:
        c = RunConfig(backend=BackendConfig(shots=2000),
                      search=SearchConfig(algorithm=algorithm, candidate_scope=scope, threshold=0.005,
                                          kl_tol=0.001, n_epochs=3, max_greedy_depth=3))
        qx = QuantumXct(c)
        qx.encode(_synth(3, 1, 0), _synth(3, 2, 0), _synth(3, 1, .3), _synth(3, 2, .3))
        cands = qx.screen_candidates()
        qx.search_topology(cands)
        assert set(qx.result.topology) <= set(cands)
        n1 = 3
        assert all(((a < n1) == (b < n1)) == (scope == "intra") for a, b in qx.result.topology)
    finally:
        from qxct import circuits, analysis
        circuits.score_circuit_kl_divergences = prev; analysis.score_circuit_kl_divergences = prev
