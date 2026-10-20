"""
qxct.baselines
==============
Baseline-method benchmark (Benchmark B): compare QuantumXCT against two
classical/tensor baselines on the *same* KL metric.

All three methods are scored the same way QuantumXCT scores itself: for each cell
register independently, the KL divergence between the method's predicted
co-culture (interacting) distribution and the true co-culture marginal, summed
over the two registers (KL_cell1 + KL_cell2). We do not build a joint
distribution across the two cell populations — they are different populations —
exactly as in the quantum pipeline.

Methods
-------
* QuantumXCT           — the hybrid quantum pipeline (interpretable gene network,
                         conditioned on the basal state, scored by sampling the
                         circuit exactly as it would run on hardware).
* Optimal Transport    — entropic Sinkhorn transport basal -> co (POT), with a
                         barycentric pushforward binned back to the bitstring grid
                         as its predicted distribution. Conditioned on basal.
* TN transform         — a learned matrix-product operator (quimb) applied to the
                         basal amplitude state: |psi_pred> = MPO @ |psi_basal>.
                         The fair classical analog of the QuantumXCT circuit —
                         conditioned on the basal state, capacity set by bond_dim.
* Born machine         — a tensor-network (MPS) Born machine (quimb) whose Born
                         probabilities |psi(x)|^2 are fit to the co-culture target
                         directly (NOT conditioned on basal — free fit; included as
                         an upper-bound/oracle reference).

For each method we report: the summed KL, the number of parameters, wall-clock
runtime, whether the method is conditioned on the basal state, and whether it
yields an interpretable interaction network.

Note on scoring: QuantumXCT is scored by sampling its circuit (nshots) because a
real quantum device returns samples, not a statevector — so its KL carries a small
amount of shot noise, while the classical baselines are computed on exact
probability vectors. This is kept deliberately (it reflects real hardware) rather
than "fixed" by reading QuantumXCT's statevector.

Optimization policy (fairness)
------------------------------
A natural question is whether every method should use the *same* optimizer and
the same max-iterations. It should not, and forcing that would REDUCE fairness,
because the methods optimize genuinely different landscapes:

  * QuantumXCT tunes a handful of CRX ANGLES against a SAMPLED (shot-noisy) KL,
    so it uses a gradient-free optimizer (COBYLA, from RunConfig.optimizer).
    A gradient method there would chase shot noise.
  * The Born machine and TN transform tune dozens-to-hundreds of smooth TENSOR
    entries against an EXACT (deterministic) probability objective, so they use
    a quasi-Newton method (L-BFGS-B). COBYLA on that many smooth parameters
    would badly under-optimize them and make them look artificially weak.

The fairness principle that actually applies is therefore "each method optimized
to best-effort convergence with matched, explicit, reproducible settings" — not
"identical optimizer". Concretely:

  * every method is seeded (seed=42 by default) for reproducibility;
  * the optimizer, max-iterations and (for OT) the entropic reg are EXPLICIT
    arguments with sensible per-method defaults, not hidden constants, so they
    can be overridden or swept;
  * OT has no such optimizer at all — Sinkhorn is a fixed-point solve set by
    `reg`, so it takes no maxiter/method knob.

Note the sampled-vs-exact asymmetry above puts QuantumXCT at a SLIGHT
DISADVANTAGE (its KL carries shot noise the baselines do not), which is kept on
purpose because it reflects how the circuit actually runs on hardware.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .circuits import calculate_kl_divergence
from .config import RunConfig

# ----------------------------------------------------------------------------
# Framing caption for Benchmark B (printed next to the tables so the
# interpretation travels with the numbers).
# ----------------------------------------------------------------------------
BENCHMARK_B_CAPTION = (
    "Benchmark B is a KL-at-matched-budget comparison. Optimal Transport and the "
    "Born machine are descriptive lower bounds on achievable KL, not mechanistic "
    "competitors: they map distributions but yield no edges and no directions (see "
    "the 'interpretable network' column). QuantumXCT's claim is not 'lowest KL' but "
    "reaching comparable KL while ALSO returning a directed, interpretable gene "
    "network. Note the two OT views reported together: the full (uncapped) solve in "
    "compare_baselines() is OT's best case, while benchmark_budget_matched() caps OT "
    "to the top-k couplings at QuantumXCT's gate budget."
)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def counts_to_prob(counts: dict, n_bits: int) -> Dict[str, float]:
    """Normalize a bitstring-count dict to a full 2**n_bits probability dict."""
    total = sum(counts.values())
    states = [format(i, f"0{n_bits}b") for i in range(2 ** n_bits)]
    if total <= 0:
        return {s: 0.0 for s in states}
    return {s: counts.get(s, 0) / total for s in states}


def _n_bits_of(counts: dict) -> int:
    return len(next(iter(counts)))


# ----------------------------------------------------------------------------
# Optimal Transport baseline (POT / entropic Sinkhorn)
# ----------------------------------------------------------------------------
def ot_sinkhorn_baseline(
    basal_counts: dict, co_counts: dict, reg: float = 0.05, max_couplings: Optional[int] = None
) -> dict:
    """Entropic OT (Sinkhorn) transporting basal -> co, scored by KL to co.

    The transport plan's barycentric map is applied to the basal support and
    binned to the nearest bitstring, giving a predicted co distribution which is
    KL-compared to the true co marginal. Not an interpretable network.

    max_couplings: if set, keep only the top-k largest transport-plan entries
        (row-renormalized) before the barycentric pushforward. This caps OT's
        effective complexity to k couplings so it can be compared at the same
        budget as QuantumXCT's gate count. ``params`` then reports k.
    """
    import ot

    n = _n_bits_of(co_counts)
    states = [format(i, f"0{n}b") for i in range(2 ** n)]
    coords = np.array([[int(b) for b in s] for s in states], dtype=float)

    p = np.array([basal_counts.get(s, 0) for s in states], dtype=float)
    q = np.array([co_counts.get(s, 0) for s in states], dtype=float)

    t0 = time.time()
    ps = p + 1e-12; ps /= ps.sum()
    qs = q + 1e-12; qs /= qs.sum()
    C = np.abs(coords[:, None, :] - coords[None, :, :]).sum(-1)   # Hamming cost
    if C.max() > 0:
        C = C / C.max()
    plan = ot.sinkhorn(ps, qs, C, reg)

    n_params = int(plan.size)
    if max_couplings is not None and max_couplings < plan.size:
        flat = plan.ravel()
        keep = np.argsort(flat)[-max_couplings:]
        mask = np.zeros_like(flat); mask[keep] = 1.0
        plan = (flat * mask).reshape(plan.shape)
        n_params = int(max_couplings)

    # barycentric pushforward (row-normalized), binned to nearest bitstring
    row = plan.sum(1)
    pred = {s: 0.0 for s in states}
    for i, s in enumerate(states):
        if row[i] <= 0:                      # this source point lost all couplings
            b = s                            # leave it in place (no transport)
        else:
            T_i = (plan[i] @ coords) / row[i]
            b = "".join("1" if v >= 0.5 else "0" for v in T_i)
        pred[b] += ps[i]

    kl = calculate_kl_divergence(pred, counts_to_prob(co_counts, n))
    runtime = time.time() - t0
    return {"kl": kl, "params": n_params, "runtime": runtime, "pred": pred}


# ----------------------------------------------------------------------------
# Tensor-network Born machine baseline (quimb MPS)
# ----------------------------------------------------------------------------
def born_machine_baseline(
    co_counts: dict, bond_dim: int = 2, maxiter: int = 400, seed: int = 42,
    optimizer: str = "L-BFGS-B"
) -> dict:
    """MPS Born machine fit to the co-culture target, scored by KL.

    A matrix-product state |psi> of the given bond dimension is optimized so that
    its Born probabilities |psi(x)|^2 match the co-culture marginal. Reports the
    achieved KL and the number of (real) MPS parameters. Not an interpretable
    network. Higher bond_dim -> more capacity (bond_dim large enough reproduces
    the target exactly).
    """
    import quimb.tensor as qtn
    from scipy.optimize import minimize

    n = _n_bits_of(co_counts)
    states = [format(i, f"0{n}b") for i in range(2 ** n)]
    qdist = counts_to_prob(co_counts, n)

    t0 = time.time()
    mps = qtn.MPS_rand_state(n, bond_dim=bond_dim, phys_dim=2, dtype=float, seed=seed)
    shapes = [a.shape for a in mps.arrays]
    sizes = [int(np.prod(s)) for s in shapes]
    nparam = int(sum(sizes))
    x0 = np.concatenate([np.asarray(a).ravel() for a in mps.arrays])

    def _probs(x):
        arrs, k = [], 0
        for sh, sz in zip(shapes, sizes):
            arrs.append(x[k:k + sz].reshape(sh)); k += sz
        vec = qtn.MatrixProductState(arrs).to_dense().flatten().real
        pr = vec ** 2
        s = pr.sum()
        return pr / s if s > 0 else pr

    def _obj(x):
        pr = _probs(x)
        return calculate_kl_divergence({states[i]: pr[i] for i in range(len(states))}, qdist)

    res = minimize(_obj, x0, method=optimizer, options={"maxiter": maxiter})
    runtime = time.time() - t0
    return {"kl": float(res.fun), "params": nparam, "runtime": runtime,
            "bond_dim": bond_dim}


# ----------------------------------------------------------------------------
# Tensor-network TRANSFORM baseline (basal -> learned MPO -> target)
# ----------------------------------------------------------------------------
def tn_transform_baseline(
    basal_counts: dict, co_counts: dict, bond_dim: int = 2, maxiter: int = 400, seed: int = 42,
    optimizer: str = "L-BFGS-B"
) -> dict:
    """Learned tensor-network transform of the basal state toward the co target.

    Unlike the free-fit Born machine, this is *conditioned on the basal state*: it
    encodes the basal distribution as an amplitude state |psi_basal> (L1 encoding,
    amplitude = sqrt(freq)), applies a matrix-product operator (MPO) of the given
    bond dimension, and scores KL(|MPO @ psi_basal|^2 || co). This is the direct
    classical analog of QuantumXCT's entangling circuit (a transform from initial
    to target), but as a dense MPO with no interpretable gene network.

    Capacity is set by ``bond_dim``: bond_dim=1 is a product (single-site) transform
    that cannot create correlations (stays lossy); larger bond_dim adds capacity and,
    at small register sizes, can drive KL -> 0.
    """
    import quimb.tensor as qtn
    from scipy.optimize import minimize

    n = _n_bits_of(co_counts)
    states = [format(i, f"0{n}b") for i in range(2 ** n)]
    pb = np.array([basal_counts.get(s, 0) for s in states], dtype=float)
    pb = pb / pb.sum() if pb.sum() > 0 else pb
    v0 = np.sqrt(pb)                                  # basal amplitude state
    qdist = counts_to_prob(co_counts, n)

    t0 = time.time()
    mpo = qtn.MPO_rand(n, bond_dim, phys_dim=2, dtype=float, seed=seed)
    shapes = [a.shape for a in mpo.arrays]
    sizes = [int(np.prod(s)) for s in shapes]
    nparam = int(sum(sizes))
    x0 = np.concatenate([np.asarray(a).ravel() for a in mpo.arrays])

    def _dense(x):
        arrs, k = [], 0
        for sh, sz in zip(shapes, sizes):
            arrs.append(x[k:k + sz].reshape(sh)); k += sz
        return qtn.MatrixProductOperator(arrs).to_dense()

    def _obj(x):
        psi = _dense(x) @ v0
        s = np.linalg.norm(psi)
        if s == 0:
            return 50.0
        pr = (psi / s) ** 2
        return calculate_kl_divergence({states[i]: pr[i] for i in range(len(states))}, qdist)

    res = minimize(_obj, x0, method=optimizer, options={"maxiter": maxiter})
    return {"kl": float(res.fun), "params": nparam, "runtime": time.time() - t0,
            "bond_dim": bond_dim}


# ----------------------------------------------------------------------------
# QuantumXCT method (per-register KL, as the pipeline already reports)
# ----------------------------------------------------------------------------
def quantumxct_baseline(
    config: RunConfig,
    counts_ct1_basal: dict, counts_ct2_basal: dict,
    counts_ct1_inter: dict, counts_ct2_inter: dict,
    gene_list: Optional[List[str]] = None,
) -> dict:
    """Run the QuantumXCT pipeline and report its summed per-register KL."""
    from .model import QuantumXct

    t0 = time.time()
    qx = QuantumXct(config, gene_list=gene_list)
    qx.encode(counts_ct1_basal, counts_ct2_basal, counts_ct1_inter, counts_ct2_inter)
    qx.screen_candidates()
    if config.search.candidate_scope == "intra":
        qx.search_topology_grn_sequential()
    else:
        qx.search_topology()
    qx.optimize_angles()
    runtime = time.time() - t0
    return {"kl": float(qx.result.kl_sum), "params": len(qx.result.topology),
            "runtime": runtime, "topology": qx.result.topology, "qx": qx}


# ----------------------------------------------------------------------------
# The Benchmark-B comparison
# ----------------------------------------------------------------------------
def compare_baselines(
    config: RunConfig,
    counts_ct1_basal: dict, counts_ct2_basal: dict,
    counts_ct1_inter: dict, counts_ct2_inter: dict,
    gene_list: Optional[List[str]] = None,
    ot_reg: float = 0.05,
    tn_bond_dim: int = 2,
    born_bond_dim: int = 2,
) -> dict:
    """Compare QuantumXCT vs OT vs TN transform vs Born machine on the same KL.

    Every method is fit per cell register and its KLs summed, to match
    QuantumXCT's KL_cell1 + KL_cell2. The 'conditioned on basal' column marks the
    methods that model the basal -> co transition (QuantumXCT, OT, TN transform)
    versus the Born machine, which free-fits the target directly. Returns a summary
    DataFrame plus the raw per-method detail.
    """
    detail = {}

    # --- QuantumXCT (topology search + angle optimization, sampled) ---
    detail["QuantumXCT"] = quantumxct_baseline(
        config, counts_ct1_basal, counts_ct2_basal,
        counts_ct1_inter, counts_ct2_inter, gene_list)

    # --- Optimal Transport (per register, summed) ---
    ot1 = ot_sinkhorn_baseline(counts_ct1_basal, counts_ct1_inter, reg=ot_reg)
    ot2 = ot_sinkhorn_baseline(counts_ct2_basal, counts_ct2_inter, reg=ot_reg)
    detail["Optimal Transport"] = {
        "kl": ot1["kl"] + ot2["kl"], "params": ot1["params"] + ot2["params"],
        "runtime": ot1["runtime"] + ot2["runtime"], "per_register": (ot1, ot2)}

    # --- TN transform: basal -> learned MPO -> co (per register, summed) ---
    tn1 = tn_transform_baseline(counts_ct1_basal, counts_ct1_inter, bond_dim=tn_bond_dim, seed=config.seed)
    tn2 = tn_transform_baseline(counts_ct2_basal, counts_ct2_inter, bond_dim=tn_bond_dim, seed=config.seed)
    detail["TN transform"] = {
        "kl": tn1["kl"] + tn2["kl"], "params": tn1["params"] + tn2["params"],
        "runtime": tn1["runtime"] + tn2["runtime"], "per_register": (tn1, tn2)}

    # --- Born machine: free fit to co (per register, summed) ---
    bm1 = born_machine_baseline(counts_ct1_inter, bond_dim=born_bond_dim, seed=config.seed)
    bm2 = born_machine_baseline(counts_ct2_inter, bond_dim=born_bond_dim, seed=config.seed)
    detail["Born machine"] = {
        "kl": bm1["kl"] + bm2["kl"], "params": bm1["params"] + bm2["params"],
        "runtime": bm1["runtime"] + bm2["runtime"], "per_register": (bm1, bm2)}

    conditioned = {"QuantumXCT": "yes", "Optimal Transport": "yes",
                   "TN transform": "yes", "Born machine": "no"}
    interpretable = {"QuantumXCT": "yes", "Optimal Transport": "no",
                     "TN transform": "no", "Born machine": "no"}
    param_kind = {"QuantumXCT": "CRX gates",
                  "Optimal Transport": "transport-plan entries",
                  "TN transform": f"MPO tensor params (D={tn_bond_dim})",
                  "Born machine": f"MPS tensor params (D={born_bond_dim})"}

    rows = []
    for method in ["QuantumXCT", "Optimal Transport", "TN transform", "Born machine"]:
        d = detail[method]
        rows.append({
            "Method": method,
            "KL (cell1+cell2)": round(float(d["kl"]), 4),
            "# params": int(d["params"]),
            "param type": param_kind[method],
            "runtime (s)": round(float(d["runtime"]), 3),
            "conditioned on basal": conditioned[method],
            "interpretable network": interpretable[method],
        })
    return {"summary": pd.DataFrame(rows), "detail": detail,
            "caption": BENCHMARK_B_CAPTION}


# ----------------------------------------------------------------------------
# Budget-matched comparison (fair: baselines capped to QuantumXCT's gate budget)
# ----------------------------------------------------------------------------
def benchmark_budget_matched(
    config: RunConfig,
    counts_ct1_basal: dict, counts_ct2_basal: dict,
    counts_ct1_inter: dict, counts_ct2_inter: dict,
    gene_list: Optional[List[str]] = None,
    ot_reg: float = 0.05,
    tn_bond_dims=(1,),
    born_bond_dims=(1,),
    ot_multiples=(1,),
    qx_best_effort: bool = True,
    qx_max_greedy_depth: Optional[int] = None,
) -> dict:
    """Fair, budget-matched Benchmark B.

    QuantumXCT is run to discover its topology; its gate count N sets the
    complexity budget. The classical baselines are then evaluated at their
    MINIMUM capacity by default (not padded with larger settings):

      - Born machine  : MPS at bond_dim = 1 (its smallest form).
      - TN transform  : MPO at bond_dim = 1 (its smallest form).
      - Optimal Transport : restricted to the top (m * N) couplings per register.

    (Pass longer tuples to tn_bond_dims / born_bond_dims / ot_multiples if you do
    want a capacity sweep.)

    qx_best_effort (default True): run QuantumXCT with no KL early-stop
    (kl_tol = 0, multi-epoch) so it keeps adding gates until KL stops improving,
    then keep the best — instead of stopping early at a few gates. This lets
    QuantumXCT compete on KL at a budget comparable to the tensor networks' minimum
    parameter footprint. Note: tensor networks have a floor (bond_dim=1) that may
    still exceed a very small QuantumXCT circuit — that parameter-efficiency gap is
    itself a result.

    Returns the QuantumXCT budget, the sweep DataFrame, and a matched-only summary.
    """
    # --- QuantumXCT: discover the budget (best-effort = no KL early stop) ---
    cfg_q = config
    if qx_best_effort:
        import dataclasses
        se = dataclasses.replace(config.search, kl_tol=0.0)
        if qx_max_greedy_depth is not None:
            se = dataclasses.replace(se, max_greedy_depth=qx_max_greedy_depth)
        cfg_q = dataclasses.replace(config, search=se)
    q = quantumxct_baseline(cfg_q, counts_ct1_basal, counts_ct2_basal,
                            counts_ct1_inter, counts_ct2_inter, gene_list)
    N = int(q["params"])                       # number of CRX gates
    budget = max(N, 1)

    sweep = []
    sweep.append({"Method": "QuantumXCT", "setting": f"{N} gates",
                  "KL (c1+c2)": round(float(q["kl"]), 4), "# params": N,
                  "within budget?": "yes", "conditioned on basal": "yes",
                  "interpretable network": "yes", "runtime (s)": round(float(q["runtime"]), 3)})

    def _sum_two(fn):
        r1 = fn(counts_ct1_basal, counts_ct1_inter)
        r2 = fn(counts_ct2_basal, counts_ct2_inter)
        return r1["kl"] + r2["kl"], r1["params"] + r2["params"], r1["runtime"] + r2["runtime"]

    # --- TN transform sweep (conditioned on basal) ---
    for D in tn_bond_dims:
        kl, pr, rt = _sum_two(lambda b, c: tn_transform_baseline(b, c, bond_dim=D, seed=config.seed))
        sweep.append({"Method": "TN transform", "setting": f"bond_dim={D}",
                      "KL (c1+c2)": round(float(kl), 4), "# params": int(pr),
                      "within budget?": "yes" if pr <= budget else "no",
                      "conditioned on basal": "yes", "interpretable network": "no",
                      "runtime (s)": round(float(rt), 3)})

    # --- Born machine sweep (free fit, not conditioned) ---
    for D in born_bond_dims:
        kl1 = born_machine_baseline(counts_ct1_inter, bond_dim=D, seed=config.seed)
        kl2 = born_machine_baseline(counts_ct2_inter, bond_dim=D, seed=config.seed)
        pr = kl1["params"] + kl2["params"]
        sweep.append({"Method": "Born machine", "setting": f"bond_dim={D}",
                      "KL (c1+c2)": round(float(kl1["kl"] + kl2["kl"]), 4), "# params": int(pr),
                      "within budget?": "yes" if pr <= budget else "no",
                      "conditioned on basal": "no", "interpretable network": "no",
                      "runtime (s)": round(float(kl1["runtime"] + kl2["runtime"]), 3)})

    # --- Optimal Transport: restricted to top (m*N) couplings per register, + native ---
    for m in ot_multiples:
        k = m * budget
        kl, pr, rt = _sum_two(lambda b, c: ot_sinkhorn_baseline(b, c, reg=ot_reg, max_couplings=k))
        sweep.append({"Method": "Optimal Transport", "setting": f"top-{k}/register",
                      "KL (c1+c2)": round(float(kl), 4), "# params": int(pr),
                      "within budget?": "yes" if pr <= budget else "no",
                      "conditioned on basal": "yes", "interpretable network": "no",
                      "runtime (s)": round(float(rt), 3)})

    sweep_df = pd.DataFrame(sweep)

    # matched summary: per method, the setting with params closest to the budget
    matched = []
    for method in ["QuantumXCT", "Optimal Transport", "TN transform", "Born machine"]:
        sub = sweep_df[sweep_df["Method"] == method].copy()
        sub["dist"] = (sub["# params"] - budget).abs()
        best = sub.sort_values(["dist"]).iloc[0]
        matched.append({k: best[k] for k in
                        ["Method", "setting", "KL (c1+c2)", "# params",
                         "within budget?", "conditioned on basal", "interpretable network"]})

    return {"quantumxct_gates": N, "param_budget": budget,
            "topology": q["topology"], "sweep": sweep_df,
            "summary": pd.DataFrame(matched),
            "caption": BENCHMARK_B_CAPTION}
