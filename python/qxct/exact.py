"""
qxct.exact
==========
Exact (shot-noise-free) evaluation of QuantumXCT circuits, plus a Qiskit-backed
analogue for cross-checking and device-style sampling.

Why exact scoring
-----------------
Contribution analyses (sequential / isolated / leave-one-out KL changes) are
evaluated with *exact* output probabilities. With a finite number of shots a
gate whose angle is ~0 (an identity) can still be assigned a non-zero KL change
purely from sampling noise, which is what made two near-identity gates appear to
"contribute" in the original Table 2. Exact probabilities remove that artifact.

Two equivalent scoring paths
----------------------------
* :class:`ExactScorer` -- a fast, dependency-light NumPy evaluator. It evolves the
  amplitude vector directly (no circuit object, no state-preparation synthesis),
  which is what makes 10+10-gene problems feasible. This is the default used by
  the topology search when :func:`use_exact_scoring` is enabled.
* :class:`QiskitScorer` -- the Qiskit analogue. It builds the actual
  ``QuantumCircuit`` (``StatePreparation`` of each cell-type register followed by
  the CRX topology) and reads the marginals either from the exact ``Statevector``
  or from sampled counts on a backend (``AerSimulator`` by default). Use it to
  validate the NumPy path against Qiskit (:func:`cross_check_exact`) or to score
  with realistic shot noise on a simulator.

Conventions (shared by both paths and by the rest of the package)
-----------------------------------------------------------------
* register 1 (cell type 1) occupies qubits ``0..n1-1``, register 2 the rest;
* the global state vector is ``kron(amp2, amp1)`` -- qubit 0 is the
  least-significant bit, identical to Qiskit's little-endian ordering;
* a gate ``(c, t)`` is ``CRX(theta)`` with control ``c`` and target ``t``;
* the cost is ``KL(P_model || Q_target)`` per register, summed over the two
  registers, with the same epsilon floor (``1e-9``) as
  ``circuits.calculate_kl_divergence``.

The module is additive: nothing in ``circuits.py`` changes behavior unless the
caller opts in via :func:`use_exact_scoring`, so all previously saved results
remain reproducible.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

EPS = 1e-9


# ----------------------------------------------------------------------------
# state helpers
# ----------------------------------------------------------------------------
def dict_to_vec(d: dict) -> Tuple[np.ndarray, int]:
    """Convert a bitstring->amplitude dict to a dense, unit-norm state vector.

    Args:
        d: Mapping from bitstring keys to amplitudes. Keys are MSB-first with
            qubit 0 as the rightmost character (the package/Qiskit convention),
            so key ``"10"`` is index 2. All keys must have the same length.
            Values may be real counts/frequencies or complex amplitudes; the
            result is L2-normalized regardless.

    Returns:
        ``(v, n)`` where ``v`` is a length-``2**n`` complex NumPy vector
        normalized to unit L2 norm (unchanged if the input norm is 0), and
        ``n`` is the number of qubits inferred from the key length.
    """
    n = len(next(iter(d)))
    v = np.zeros(2 ** n, dtype=complex)
    for k, a in d.items():
        v[int(k, 2)] = complex(a)
    nrm = np.linalg.norm(v)
    return (v / nrm if nrm > 0 else v), n


def product_state(amp1: dict, amp2: dict) -> Tuple[np.ndarray, int, int]:
    """Tensor the two per-register amplitude dicts into one global state vector.

    Args:
        amp1: Register-1 (cell type 1) amplitude dict (see :func:`dict_to_vec`).
        amp2: Register-2 (cell type 2) amplitude dict.

    Returns:
        ``(psi, n1, n2)`` where ``psi = kron(v2, v1)`` is the length-``2**(n1+n2)``
        global state (register 1 on the low qubits), and ``n1``/``n2`` are the
        qubit counts of the two registers.
    """
    v1, n1 = dict_to_vec(amp1)
    v2, n2 = dict_to_vec(amp2)
    return np.kron(v2, v1), n1, n2


def target_probs(amp: dict) -> np.ndarray:
    """Born probabilities ``|amplitude|**2`` of a single-register target dict.

    Args:
        amp: Target amplitude dict for one register (see :func:`dict_to_vec`).

    Returns:
        A length-``2**n`` real probability vector that sums to 1.
    """
    v, _ = dict_to_vec(amp)
    return np.abs(v) ** 2


def apply_crx(psi: np.ndarray, n: int, control: int, target: int, theta: float) -> np.ndarray:
    """Apply ``CRX(theta)`` (Qiskit convention) to an ``n``-qubit state vector.

    ``CRX`` rotates ``target`` by ``RX(theta)`` only on the subspace where
    ``control`` is 1; the control marginal is unchanged.

    Args:
        psi: Length-``2**n`` complex state vector (little-endian, qubit 0 = LSB).
        n: Number of qubits.
        control: Control qubit index.
        target: Target qubit index (must differ from ``control``).
        theta: Rotation angle in radians.

    Returns:
        The evolved length-``2**n`` state vector (a new array; ``psi`` is not
        modified in place).
    """
    t = np.asarray(psi, dtype=complex).reshape([2] * n)   # axis k <-> qubit n-1-k
    ac, at = n - 1 - control, n - 1 - target
    c, s = np.cos(theta / 2.0), -1j * np.sin(theta / 2.0)
    idx1 = [slice(None)] * n
    idx1[ac] = 1
    sub = t[tuple(idx1)]              # control = 1 block (one axis removed)
    at_sub = at if at < ac else at - 1
    a0 = np.take(sub, 0, axis=at_sub)
    a1 = np.take(sub, 1, axis=at_sub)
    new0 = c * a0 + s * a1
    new1 = s * a0 + c * a1
    sub_new = np.stack([new0, new1], axis=at_sub)
    out = t.copy()
    out[tuple(idx1)] = sub_new
    return out.reshape(-1)


def apply_ry(psi: np.ndarray, n: int, qubit: int, theta: float) -> np.ndarray:
    """Apply a single-qubit ``RY(theta)`` to an ``n``-qubit state vector.

    Used only by the local-rotation (no-entangler) null model.

    Args:
        psi: Length-``2**n`` complex state vector (little-endian).
        n: Number of qubits.
        qubit: Qubit index to rotate.
        theta: Rotation angle in radians.

    Returns:
        The evolved length-``2**n`` state vector.
    """
    t = psi.reshape([2] * n)
    ax = n - 1 - qubit
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    a0 = np.take(t, 0, axis=ax)
    a1 = np.take(t, 1, axis=ax)
    return np.stack([c * a0 - s * a1, s * a0 + c * a1], axis=ax).reshape(-1)


def marginals(psi: np.ndarray, n1: int, n2: int) -> Tuple[np.ndarray, np.ndarray]:
    """Exact computational-basis marginals of the two registers.

    Args:
        psi: Length-``2**(n1+n2)`` global state vector (register 1 on low qubits).
        n1: Number of qubits in register 1.
        n2: Number of qubits in register 2.

    Returns:
        ``(p1, p2)`` real probability vectors of length ``2**n1`` and ``2**n2``,
        each summing to 1, obtained by tracing out the other register.
    """
    p = (np.abs(psi) ** 2).reshape(2 ** n2, 2 ** n1)   # row = reg2 index, col = reg1 index
    return p.sum(axis=0), p.sum(axis=1)


def kl(p: np.ndarray, q: np.ndarray, eps: float = EPS) -> float:
    """``KL(P || Q)`` with the package's two-sided epsilon floor.

    Args:
        p: Observed/model probability vector.
        q: Target probability vector (same length as ``p``).
        eps: Floor applied to both ``p`` and ``q`` before the log, matching
            ``circuits.calculate_kl_divergence`` so results are comparable.

    Returns:
        The scalar KL divergence ``sum p * log(p/q)`` as a Python float.
    """
    p = np.maximum(p, eps)
    q = np.maximum(q, eps)
    return float(np.sum(p * np.log(p / q)))


# ----------------------------------------------------------------------------
# main evaluator (NumPy)
# ----------------------------------------------------------------------------
class ExactScorer:
    """Exact ``KL_1 + KL_2`` for any CRX topology/angles on a fixed problem.

    The problem (initial and target states of both registers) is encoded once in
    the constructor; afterwards any topology and angle vector can be scored
    cheaply by evolving the stored initial state vector. All scoring is exact
    (shot-noise-free); the module-level :func:`sampled_kl` and :func:`shot_noise_samples` add Qiskit-native shot noise on demand.

    Args:
        init1: Register-1 initial (mono-culture) amplitude dict.
        init2: Register-2 initial amplitude dict.
        target1: Register-1 target (co-culture) amplitude dict.
        target2: Register-2 target amplitude dict.

    Attributes:
        psi0: The encoded global initial state vector.
        n1, n2, n: Register and total qubit counts.
        q1, q2: Target Born-probability vectors of the two registers.
    """

    def __init__(self, init1: dict, init2: dict, target1: dict, target2: dict):
        self.psi0, self.n1, self.n2 = product_state(init1, init2)
        self.n = self.n1 + self.n2
        self.q1 = target_probs(target1)
        self.q2 = target_probs(target2)
        # kept so a Qiskit scorer (for cross-check or sampling) can be built from
        # the same encoded problem via :meth:`qiskit`.
        self.init1, self.init2 = init1, init2
        self.target1, self.target2 = target1, target2

    def state(self, topology: Sequence[Tuple[int, int]],
              angles: Optional[Sequence[float]] = None) -> np.ndarray:
        """Evolve the encoded initial state through a CRX topology.

        Args:
            topology: Ordered list of ``(control, target)`` qubit-index pairs.
            angles: Angle (radians) for each gate; defaults to ``pi/2`` per gate
                (the topology-search value) when ``None``.

        Returns:
            The resulting length-``2**n`` global state vector.
        """
        if angles is None:
            angles = [np.pi / 2] * len(topology)
        psi = self.psi0
        for (c, t), th in zip(topology, angles):
            psi = apply_crx(psi, self.n, c, t, th)
        return psi

    def kl_parts(self, topology, angles=None) -> Tuple[float, float]:
        """Per-register exact KL.

        Args:
            topology: Ordered ``(control, target)`` pairs.
            angles: Per-gate angles (defaults to ``pi/2`` each).

        Returns:
            ``(kl1, kl2)`` -- the exact KL of register 1 and register 2 against
            their targets.
        """
        m1, m2 = marginals(self.state(topology, angles), self.n1, self.n2)
        return kl(m1, self.q1), kl(m2, self.q2)

    def kl(self, topology, angles=None) -> float:
        """Total exact loss ``KL_1 + KL_2``.

        Args:
            topology: Ordered ``(control, target)`` pairs.
            angles: Per-gate angles (defaults to ``pi/2`` each).

        Returns:
            The summed per-register KL as a float.
        """
        a, b = self.kl_parts(topology, angles)
        return a + b

    def baseline(self) -> float:
        """Loss of the bare initial state (empty topology).

        Returns:
            ``KL_1 + KL_2`` with no gates applied -- the denominator for all
            reduction percentages.
        """
        return self.kl([], [])

    # --- Qiskit view of the same problem (cross-check / sampling) ---
    def qiskit(self, mode: str = "statevector", shots: int = 5000,
               backend=None, seed: int = 42) -> "QiskitScorer":
        """A :class:`QiskitScorer` for the same encoded problem.

        Use ``mode="statevector"`` to cross-check this NumPy evaluator against
        Qiskit, or ``mode="sampler"`` to score with real shot noise.

        Args:
            mode: ``"statevector"`` (exact) or ``"sampler"`` (shot-based).
            shots: Shots per register when sampling.
            backend: Optional Qiskit backend for ``"sampler"`` mode.
            seed: Sampler seed.

        Returns:
            A :class:`QiskitScorer` bound to this problem.
        """
        return QiskitScorer(self.init1, self.init2, self.target1, self.target2,
                            mode=mode, shots=shots, backend=backend, seed=seed)

    # --- angle optimization on the exact objective ---
    def optimize_angles(self, topology, init: float = np.pi / 2, method: str = "COBYLA",
                        restarts: int = 0, seed: int = 42, bounds=(0.0, np.pi)):
        """Optimize the gate angles of a fixed topology on the exact loss.

        Args:
            topology: Ordered ``(control, target)`` pairs to optimize.
            init: Starting angle (radians) for every gate on the first restart.
            method: ``scipy.optimize.minimize`` method (default ``"COBYLA"``).
            restarts: Number of additional random-start restarts; the best
                result over all starts is returned.
            seed: Seed for the random restarts.
            bounds: ``(low, high)`` angle bounds applied to every gate.

        Returns:
            ``(angles, loss)`` -- the optimized angle list and the exact loss at
            that optimum. For an empty topology returns ``([], baseline())``.
        """
        from scipy.optimize import minimize
        if len(topology) == 0:
            return [], self.baseline()
        rng = np.random.default_rng(seed)
        best = None
        starts = [np.full(len(topology), init)] + [
            rng.uniform(bounds[0], bounds[1], len(topology)) for _ in range(restarts)]
        for x0 in starts:
            r = minimize(lambda x: self.kl(topology, x), x0, method=method,
                         bounds=[bounds] * len(topology))
            if best is None or r.fun < best.fun:
                best = r
        return list(best.x), float(best.fun)

    # --- local-only (no entangler) baseline ---
    def local_ry_baseline(self, restarts: int = 20, seed: int = 42):
        """Best loss reachable with one free ``RY`` per qubit and no two-qubit gates.

        This is the "local self-regulation" null: per-gene re-tuning with no
        interactions (``n`` free parameters). It is a reference point, not a
        competing model of communication.

        Args:
            restarts: Number of random restarts (plus one zero-start).
            seed: Seed for the random restarts.

        Returns:
            ``(phis, loss)`` -- the optimized per-qubit angles and the exact loss.
        """
        from scipy.optimize import minimize
        rng = np.random.default_rng(seed)

        def f(x):
            psi = self.psi0
            for q, th in enumerate(x):
                psi = apply_ry(psi, self.n, q, th)
            m1, m2 = marginals(psi, self.n1, self.n2)
            return kl(m1, self.q1) + kl(m2, self.q2)

        best = None
        for r in range(restarts + 1):
            x0 = np.zeros(self.n) if r == 0 else rng.uniform(-np.pi, np.pi, self.n)
            res = minimize(f, x0, method="L-BFGS-B")
            if best is None or res.fun < best.fun:
                best = res
        return list(best.x), float(best.fun)


# ----------------------------------------------------------------------------
# exact contribution tables
# ----------------------------------------------------------------------------
def exact_contribution_table(scorer: ExactScorer, topology, angles,
                             gene_list: Optional[List[str]] = None,
                             nshots: int = 5000, n_rep: int = 200, seed: int = 42):
    """Per-gate contribution table computed with exact probabilities.

    For each gate three complementary contribution measures are reported, all as
    a share of the total KL reduction (``baseline - final``):

    * cumulative / sequential -- the loss after adding gates in order (depends on
      the order);
    * isolated -- the reduction achieved by that one gate alone;
    * leave-one-out (LOO) -- the loss increase when that gate is removed from the
      full circuit (order-independent; need not sum to 100% because gates interact).

    The shot-noise standard deviation of the final loss at ``nshots`` is also
    returned, so contributions below ~2 sigma can be flagged as unresolvable from
    sampling.

    Args:
        scorer: An :class:`ExactScorer` for the encoded problem.
        topology: Ordered ``(control, target)`` pairs of the final circuit.
        angles: Optimized angle for each gate.
        gene_list: Optional gene names for labeling rows; falls back to ``qC->qT``.
        nshots: Shot count for the Qiskit-sampled shot-noise estimate (the
            contributions themselves are always exact, never sampled).
        n_rep: Number of Qiskit-sampled replicates used to estimate the sd.
        seed: Base seed for the shot-noise replicates.

    Returns:
        ``(df, meta)`` where ``df`` is a pandas ``DataFrame`` with one row per gate
        (step, gate label, theta, cumulative/step/isolated/LOO KL and their
        percentages) and ``meta`` is a dict with ``baseline_kl``, ``final_kl``,
        ``total_reduction``, ``pct_reduction``, ``shot_sd_final_kl`` and ``nshots``.
    """
    import pandas as pd
    lab = (lambda c, t: f"{gene_list[c]} -> {gene_list[t]}") if gene_list else (lambda c, t: f"q{c}->q{t}")
    base = scorer.baseline()
    full = scorer.kl(topology, angles)
    total_red = base - full
    # shot-noise sd of the final loss, from Qiskit-sampled realizations of the
    # actual circuit (joint measurement, then marginalize) -- not a NumPy fake.
    sd_full = float(np.std(shot_noise_samples(
        scorer.init1, scorer.init2, scorer.target1, scorer.target2,
        topology, angles, shots=nshots, n_rep=n_rep, seed=seed)))
    rows, prev = [], base
    for i, ((c, t), th) in enumerate(zip(topology, angles), start=1):
        k_cum = scorer.kl(topology[:i], angles[:i])
        k_iso = scorer.kl([(c, t)], [th])
        loo_t = list(topology[:i - 1]) + list(topology[i:])
        loo_a = list(angles[:i - 1]) + list(angles[i:])
        k_loo = scorer.kl(loo_t, loo_a)
        rows.append({
            "step": i, "gate": lab(c, t), "theta_rad": th,
            "KL_cumulative": k_cum, "dKL_step": k_cum - prev,
            "pct_of_total_reduction_step": 100 * (prev - k_cum) / total_red if total_red > 0 else np.nan,
            "KL_isolated": k_iso, "isolated_reduction": base - k_iso,
            "KL_leave_one_out": k_loo, "LOO_increase": k_loo - full,
            "pct_LOO_of_total_reduction": 100 * (k_loo - full) / total_red if total_red > 0 else np.nan,
        })
        prev = k_cum
    df = pd.DataFrame(rows)
    return df, {"baseline_kl": base, "final_kl": full, "total_reduction": total_red,
                "pct_reduction": 100 * total_red / base if base > 0 else np.nan,
                "shot_sd_final_kl": sd_full, "nshots": nshots}


# ----------------------------------------------------------------------------
# opt-in: make the package's search functions score exactly (Qiskit statevector)
# ----------------------------------------------------------------------------
def _fast_statevector(qc):
    """Statevector of a QuantumXCT circuit without synthesizing the state prep.

    Leading ``Initialize`` / ``StatePreparation`` instructions that act on
    contiguous qubit blocks (the per-cell-type encodings) are replaced by the
    product of their target vectors; only the remaining gates (the CRX topology)
    are evolved. This gives the same state as ``Statevector(qc)`` -- ``Initialize``
    resets fresh ``|0>`` qubits, so it is a pure state preparation there -- but
    avoids decomposing a ``2**n``-amplitude isometry, which is what makes
    10+10-gene circuits feasible. Falls back to ``Statevector(qc)`` for any other
    circuit shape.

    Args:
        qc: A Qiskit ``QuantumCircuit`` (measurements already removed).

    Returns:
        A Qiskit ``Statevector`` of the full circuit.
    """
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    blocks, rest_start = [], 0
    for k, inst in enumerate(qc.data):
        name = inst.operation.name
        if name in ("initialize", "state_preparation"):
            qs = [qc.find_bit(q).index for q in inst.qubits]
            params = inst.operation.params
            if not qs == list(range(qs[0], qs[0] + len(qs))) or isinstance(params[0], str) or len(params) != 2 ** len(qs):
                return Statevector(qc)
            blocks.append((qs[0], np.asarray(params, dtype=complex)))
            rest_start = k + 1
        elif name == "barrier":
            continue
        else:
            break
    if not blocks:
        return Statevector(qc)
    blocks.sort(key=lambda b: b[0])
    pos, vec = 0, np.array([1.0 + 0j])
    for start, v in blocks:
        if start != pos:
            return Statevector(qc)
        vec = np.kron(v, vec)                     # later block = higher qubits
        pos += int(np.log2(v.size))
    if pos != qc.num_qubits:
        return Statevector(qc)
    rest = QuantumCircuit(*qc.qregs)
    for inst in qc.data[rest_start:]:
        if inst.operation.name != "barrier":
            rest.append(inst.operation, inst.qubits, inst.clbits)
    return Statevector(vec / np.linalg.norm(vec)).evolve(rest)


def exact_score_circuit(circuit, target1, target2, nshots=None, backend=None,
                        error_mitigation=False, norm=None):
    """Exact-statevector drop-in for ``circuits.score_circuit_kl_divergences``.

    Reads the circuit's exact statevector (via :func:`_fast_statevector`) instead
    of sampling, and returns the per-register KL against the targets. The
    signature matches the sampled scorer it replaces, so ``nshots``, ``backend``
    and ``error_mitigation`` are accepted and ignored.

    Args:
        circuit: A QuantumXCT ``QuantumCircuit`` with two classical registers
            (its final measurements are removed internally).
        target1: Register-1 target state (dict/list/array; see
            ``circuits._process_target_state_input``).
        target2: Register-2 target state.
        nshots: Ignored (present for signature compatibility).
        backend: Ignored.
        error_mitigation: Ignored.
        norm: Encoding convention passed to ``_process_target_state_input``
            (``None`` means the target is already amplitudes).

    Returns:
        ``(kl1, kl2)`` -- the exact per-register KL divergences.
    """
    from qiskit.quantum_info import Statevector
    from .circuits import _process_target_state_input
    qc = circuit.remove_final_measurements(inplace=False)
    sv = _fast_statevector(qc)
    n1 = circuit.cregs[0].size
    n = circuit.num_qubits
    m1 = sv.probabilities(qargs=list(range(n1)))
    m2 = sv.probabilities(qargs=list(range(n1, n)))
    t1, _ = _process_target_state_input(target1, norm)
    t2, _ = _process_target_state_input(target2, norm)
    return kl(m1, np.abs(t1) ** 2), kl(m2, np.abs(t2) ** 2)


def use_exact_scoring(enable: bool = True):
    """Monkey-patch the search/optimization code to score exactly.

    When enabled, ``circuits.score_circuit_kl_divergences`` and
    ``analysis.score_circuit_kl_divergences`` are replaced by
    :func:`exact_score_circuit`, so the topology search and angle optimization use
    exact probabilities instead of 5000-shot estimates.

    Args:
        enable: ``True`` to switch on exact scoring, ``False`` is a no-op that
            still returns the current scorer.

    Returns:
        The previous ``score_circuit_kl_divergences`` callable, so the caller can
        restore it later (``circuits.score_circuit_kl_divergences = prev``).
    """
    from . import circuits, analysis
    prev = circuits.score_circuit_kl_divergences
    if enable:
        circuits.score_circuit_kl_divergences = exact_score_circuit
        analysis.score_circuit_kl_divergences = exact_score_circuit
    return prev


# ----------------------------------------------------------------------------
# Qiskit analogue: build the real circuit and score it two ways
# ----------------------------------------------------------------------------
# The functions above evolve amplitude vectors with NumPy, which is fast but
# bypasses Qiskit entirely. The helpers below do the same scoring through an
# actual QuantumCircuit, so the two paths can be cross-checked and so the
# circuit can be sampled on a simulator with realistic shot noise. All Qiskit
# imports are local, so importing qxct.exact never requires Qiskit.
# ----------------------------------------------------------------------------
def build_qiskit_circuit(init1: dict, init2: dict,
                         topology: Sequence[Tuple[int, int]],
                         angles: Optional[Sequence[float]] = None,
                         measure: bool = False):
    """Build the QuantumXCT circuit as a Qiskit ``QuantumCircuit``.

    Register 1 is prepared on qubits ``0..n1-1`` and register 2 on the rest with
    ``StatePreparation`` (a true unitary, no resets), followed by the CRX
    topology. This matches the convention of :class:`ExactScorer` exactly, so the
    resulting statevector equals ``ExactScorer.state(...)`` up to numerical
    precision.

    Args:
        init1: Register-1 initial amplitude dict.
        init2: Register-2 initial amplitude dict.
        topology: Ordered ``(control, target)`` qubit-index pairs.
        angles: Per-gate angles (defaults to ``pi/2`` each when ``None``).
        measure: If ``True``, append two classical registers ``c_ct1``/``c_ct2``
            and measure register 1 and register 2 into them (needed for sampling).

    Returns:
        A Qiskit ``QuantumCircuit`` over ``n1 + n2`` qubits.
    """
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit.circuit.library import StatePreparation
    v1, n1 = dict_to_vec(init1)
    v2, n2 = dict_to_vec(init2)
    if angles is None:
        angles = [np.pi / 2] * len(topology)
    qr = QuantumRegister(n1 + n2, "q")
    qc = QuantumCircuit(qr)
    qc.append(StatePreparation(v1), range(n1))
    qc.append(StatePreparation(v2), range(n1, n1 + n2))
    for (c, t), th in zip(topology, angles):
        qc.crx(float(th), c, t)
    if measure:
        c1 = ClassicalRegister(n1, "c_ct1")
        c2 = ClassicalRegister(n2, "c_ct2")
        qc.add_register(c1, c2)
        qc.measure(range(n1), c1)
        qc.measure(range(n1, n1 + n2), c2)
    return qc


def qiskit_statevector_marginals(init1, init2, topology, angles=None):
    """Exact register marginals from Qiskit's ``Statevector`` (no sampling).

    This is the Qiskit analogue of ``marginals(ExactScorer.state(...))``.

    Args:
        init1: Register-1 initial amplitude dict.
        init2: Register-2 initial amplitude dict.
        topology: Ordered ``(control, target)`` pairs.
        angles: Per-gate angles (defaults to ``pi/2`` each).

    Returns:
        ``(p1, p2)`` real probability vectors of length ``2**n1`` and ``2**n2``.
    """
    from qiskit.quantum_info import Statevector
    _, n1 = dict_to_vec(init1)
    _, n2 = dict_to_vec(init2)
    qc = build_qiskit_circuit(init1, init2, topology, angles, measure=False)
    sv = Statevector(qc)
    p1 = sv.probabilities(qargs=list(range(n1)))
    p2 = sv.probabilities(qargs=list(range(n1, n1 + n2)))
    return np.asarray(p1), np.asarray(p2)


def qiskit_sampled_marginals(init1, init2, topology, angles=None,
                             shots: int = 5000, backend=None, seed: int = 42):
    """Sampled register marginals with realistic shot noise, Qiskit-native.

    By default the circuit's exact ``Statevector`` is sampled with Qiskit's
    ``sample_counts`` -- a true multinomial draw of ``shots`` joint measurements
    that are then marginalized per register (so the two registers share the same
    shots, exactly as a real run would). Pass a ``backend`` (e.g. an
    ``AerSimulator`` or a device sampler) to run the measured circuit on it
    instead.

    Args:
        init1: Register-1 initial amplitude dict.
        init2: Register-2 initial amplitude dict.
        topology: Ordered ``(control, target)`` pairs.
        angles: Per-gate angles (defaults to ``pi/2`` each).
        shots: Number of shots.
        backend: Optional Qiskit backend/simulator; default samples the
            statevector directly (no backend/transpile needed).
        seed: Seed for reproducibility.

    Returns:
        ``(p1, p2)`` sampled probability vectors of length ``2**n1`` and
        ``2**n2`` (zero-filled over every outcome), each summing to 1.
    """
    _, n1 = dict_to_vec(init1)
    _, n2 = dict_to_vec(init2)

    def counts_to_probs(counts, nbits):
        v = np.zeros(2 ** nbits)
        tot = sum(counts.values()) or 1
        for bitstring, c in counts.items():
            v[int(bitstring, 2)] = c / tot
        return v

    if backend is None:
        # Native statevector sampling: joint counts, keyed MSB-first over all
        # qubits (reg 2 on the high bits, reg 1 on the low bits).
        from qiskit.quantum_info import Statevector
        qc = build_qiskit_circuit(init1, init2, topology, angles, measure=False)
        joint = Statevector(qc).sample_counts(shots, qargs=list(range(n1 + n2)))
    else:
        # Run the measured circuit on the given backend; Aer returns two
        # space-separated register fields "c_ct2 c_ct1".
        from qiskit import transpile
        qc = build_qiskit_circuit(init1, init2, topology, angles, measure=True)
        isa = transpile(qc, backend, seed_transpiler=seed)
        res = backend.run(isa, shots=shots, seed_simulator=seed).result()
        joint = {k.replace(" ", ""): v for k, v in res.get_counts().items()}
    c1, c2 = {}, {}
    for key, n in joint.items():
        k2, k1 = key[:n2], key[n2:]           # MSB-first: reg 2 high, reg 1 low
        c1[k1] = c1.get(k1, 0) + n
        c2[k2] = c2.get(k2, 0) + n
    return counts_to_probs(c1, n1), counts_to_probs(c2, n2)


def sampled_kl(init1, init2, target1, target2, topology, angles=None,
               shots: int = 5000, backend=None, seed: int = 42) -> float:
    """One shot-noisy realization of the loss, Qiskit-native.

    Replaces the old NumPy multinomial fake: the circuit is actually sampled
    (via :func:`qiskit_sampled_marginals`) and the per-register KL recomputed.

    Args:
        init1, init2: Register initial amplitude dicts.
        target1, target2: Register target amplitude dicts.
        topology: Ordered ``(control, target)`` pairs.
        angles: Per-gate angles (defaults to ``pi/2`` each).
        shots: Number of shots.
        backend: Optional Qiskit backend (default: statevector sampling).
        seed: Seed for the draw.

    Returns:
        A single sampled estimate of ``KL_1 + KL_2``.
    """
    p1, p2 = qiskit_sampled_marginals(init1, init2, topology, angles,
                                      shots=shots, backend=backend, seed=seed)
    return kl(p1, target_probs(target1)) + kl(p2, target_probs(target2))


def shot_noise_samples(init1, init2, target1, target2, topology, angles=None,
                       shots: int = 5000, n_rep: int = 200, backend=None,
                       seed: int = 42) -> List[float]:
    """``n_rep`` independent Qiskit-sampled losses, for a shot-noise estimate.

    Each replicate samples the circuit afresh with a distinct seed, so the
    spread of the returned values is the shot-noise distribution of the loss at
    ``shots`` shots. Take ``np.std(...)`` for the standard deviation and
    ``np.mean(...)`` for the (bias-included) expected sampled loss.

    Args:
        init1, init2, target1, target2: The encoded problem.
        topology: Ordered ``(control, target)`` pairs.
        angles: Per-gate angles (defaults to ``pi/2`` each).
        shots: Shots per replicate.
        n_rep: Number of replicates.
        backend: Optional Qiskit backend (default: statevector sampling).
        seed: Base seed; replicate ``i`` uses ``seed + i``.

    Returns:
        A list of ``n_rep`` sampled ``KL_1 + KL_2`` values.
    """
    return [sampled_kl(init1, init2, target1, target2, topology, angles,
                       shots=shots, backend=backend, seed=seed + i)
            for i in range(n_rep)]


class QiskitScorer:
    """Qiskit-backed analogue of :class:`ExactScorer`.

    Mirrors the :meth:`ExactScorer.kl` / :meth:`kl_parts` / :meth:`baseline` API,
    but scores by building the real ``QuantumCircuit`` and reading its marginals
    through Qiskit -- exactly from the ``Statevector`` (``mode="statevector"``) or
    from sampled counts on a simulator (``mode="sampler"``). Use it to validate
    the NumPy path or to score with realistic shot noise.

    Args:
        init1, init2: Register initial amplitude dicts.
        target1, target2: Register target amplitude dicts.
        mode: ``"statevector"`` (exact, default) or ``"sampler"`` (shot-based).
        shots: Shots per register when ``mode="sampler"``.
        backend: Optional backend for ``mode="sampler"`` (default ``AerSimulator``).
        seed: Seed for the sampler.
    """

    def __init__(self, init1, init2, target1, target2,
                 mode: str = "statevector", shots: int = 5000, backend=None, seed: int = 42):
        self.init1, self.init2 = init1, init2
        self.q1 = target_probs(target1)
        self.q2 = target_probs(target2)
        self.mode, self.shots, self.backend, self.seed = mode, shots, backend, seed

    def _marginals(self, topology, angles):
        if self.mode == "sampler":
            return qiskit_sampled_marginals(self.init1, self.init2, topology, angles,
                                            shots=self.shots, backend=self.backend, seed=self.seed)
        return qiskit_statevector_marginals(self.init1, self.init2, topology, angles)

    def kl_parts(self, topology, angles=None) -> Tuple[float, float]:
        """Per-register KL via Qiskit. See :meth:`ExactScorer.kl_parts`."""
        m1, m2 = self._marginals(topology, angles)
        return kl(m1, self.q1), kl(m2, self.q2)

    def kl(self, topology, angles=None) -> float:
        """Total loss ``KL_1 + KL_2`` via Qiskit. See :meth:`ExactScorer.kl`."""
        a, b = self.kl_parts(topology, angles)
        return a + b

    def baseline(self) -> float:
        """Loss of the bare encoded state (empty topology), via Qiskit."""
        return self.kl([], [])


def cross_check_exact(init1, init2, target1, target2, topology, angles=None,
                      atol: float = 1e-9):
    """Verify the NumPy and Qiskit statevector paths agree.

    Scores the same topology with :class:`ExactScorer` (NumPy) and
    :class:`QiskitScorer` (Qiskit statevector) and reports the difference. Useful
    as a regression check that the fast NumPy evaluator matches Qiskit.

    Args:
        init1, init2, target1, target2: The encoded problem.
        topology: Ordered ``(control, target)`` pairs.
        angles: Per-gate angles (defaults to ``pi/2`` each).
        atol: Absolute tolerance for the ``agree`` flag.

    Returns:
        A dict with ``numpy_kl``, ``qiskit_kl``, ``abs_diff`` and ``agree``
        (``abs_diff <= atol``).
    """
    ns = ExactScorer(init1, init2, target1, target2).kl(topology, angles)
    qs = QiskitScorer(init1, init2, target1, target2, mode="statevector").kl(topology, angles)
    diff = abs(ns - qs)
    return {"numpy_kl": ns, "qiskit_kl": qs, "abs_diff": diff, "agree": diff <= atol}
