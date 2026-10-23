"""
qxct.analysis
============
Benchmarking utilities for QuantumXCT topology search.

This module reproduces the classic KL-value / reduction analysis, but broken out
by the three biological search scopes and with two complementary contribution
tables:

Search scopes
-------------
* ``all``   — GRN + LR: search every candidate link (intra- and inter-cellular).
* ``intra`` — GRN only: within-cell links, optimized one cell at a time
              (cell 1 to convergence, then cell 2); LR links may not contribute.
* ``inter`` — LR only: cross-cell (ligand-receptor) links, searched over all
              combinations (still density-matrix screened).

Contribution tables (for a fixed, optimized topology)
-----------------------------------------------------
* :func:`cumulative_ablation_table` — add the gates back in order and record the
  running KL and the *cumulative* reduction from baseline (sequential / order-
  dependent view).
* :func:`isolated_contribution_table` — for each gate, its *isolated* effect
  (that gate alone vs. baseline) and its *marginal* leave-one-out effect (full
  circuit minus that gate). Order-independent view of each gate's importance.

:func:`compare_scopes` runs the full search+optimize for each scope and returns a
single summary DataFrame plus the per-scope detail.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .circuits import (
    concatenate_circuits_with_separate_measurements,
    apply_entanglement_topology,
    add_crx_gates_and_measurements_to_circuit,
    score_circuit_kl_divergences,
    Sampler,
    generate_preset_pass_manager,
    _default_backend,
    _process_target_state_input,
    get_probability_distribution,
    plot_measurement_histograms,
    analyze_and_summarize_network,
    plot_quantum_relay_network,
)
from .config import RunConfig
from .model import QuantumXct

# Occupancy-plot palette (matches the original functional notebooks):
_C_MONO = "#4682B4"   # mono / basal      — darker steel blue
_C_QXCT = "#9370DB"   # QuantumXCT output — medium purple
_C_CO   = "#4B0082"   # co-culture target — deep indigo


# ----------------------------------------------------------------------------
# Low-level KL helpers
# ----------------------------------------------------------------------------
def _kl_for_topology(circ1, circ2, topology, angles, target1, target2, nshots, norm=None) -> float:
    """KL sum for a CRX circuit with the given topology & angles."""
    base = concatenate_circuits_with_separate_measurements(circ1, circ2)
    circ = add_crx_gates_and_measurements_to_circuit(base, circ1.num_qubits, list(topology), list(angles))
    k1, k2 = score_circuit_kl_divergences(circ, target1, target2, nshots, norm=norm)
    if k1 is None or k2 is None:
        return float("inf")
    return k1 + k2


def baseline_kl(circ1, circ2, target1, target2, nshots, norm=None) -> float:
    """KL sum with no interaction gates."""
    base = concatenate_circuits_with_separate_measurements(circ1, circ2)
    circ = apply_entanglement_topology(base, circ1.num_qubits, [])
    k1, k2 = score_circuit_kl_divergences(circ, target1, target2, nshots, norm=norm)
    return (k1 + k2) if (k1 is not None and k2 is not None) else float("inf")


def _amp_to_prob(state_input, norm):
    """Amplitude dict/array -> {bitstring: probability} over all 2**n outcomes."""
    amp, n = _process_target_state_input(state_input, norm)
    return {bin(i)[2:].zfill(n): float(np.abs(amp[i]) ** 2) for i in range(2 ** n)}


def post_optimization_distributions(
    circ1, circ2, topology, angles, target1, target2, nshots=5000, backend=None, norm=None,
    basal1=None, basal2=None,
):
    """Run the POST-angle-optimization circuit once and return the measured,
    target (and optionally basal) probability distributions for each register.

    Returns a dict with keys ``sim1``/``tgt1`` (register 1, cell type 1) and
    ``sim2``/``tgt2`` (register 2, cell type 2); when ``basal1``/``basal2``
    amplitudes are supplied, ``mono1``/``mono2`` are added too. Each value is an
    ordered dict ``{bitstring: probability}`` covering all 2**n outcomes.
    """
    if backend is None:
        backend = _default_backend()

    base = concatenate_circuits_with_separate_measurements(circ1, circ2)
    circ = add_crx_gates_and_measurements_to_circuit(
        base, circ1.num_qubits, list(topology), list(angles))

    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    qc_comp = pm.run(circ.decompose())
    job = Sampler(mode=backend).run([qc_comp], shots=nshots)
    result = job.result()[0]
    counts1 = result.data.c_measure1.get_counts()
    counts2 = result.data.c_measure2.get_counts()

    nb1 = circ.cregs[0].size
    nb2 = circ.cregs[1].size if len(circ.cregs) > 1 else 0
    sim1 = get_probability_distribution(counts1, nb1, nshots)
    sim2 = get_probability_distribution(counts2, nb2, nshots)

    # Target (co-culture) amplitudes -> probability distributions (|amp|**2).
    out = {"sim1": sim1, "tgt1": _amp_to_prob(target1, norm),
           "sim2": sim2, "tgt2": _amp_to_prob(target2, norm)}
    # Basal (mono-culture) initial-state amplitudes -> probability distributions.
    if basal1 is not None:
        out["mono1"] = _amp_to_prob(basal1, norm)
    if basal2 is not None:
        out["mono2"] = _amp_to_prob(basal2, norm)
    return out


def plot_scope_histograms(
    res: Dict, nshots: int = 5000, backend=None, norm=None,
    title_prefix: str = "", figure_save_name: Optional[str] = None,
    show: bool = True, figsize: Tuple[int, int] = (13, 5),
):
    """Plot the POST-angle-optimization measured distribution against the
    co-culture target, per register, for a single scope result from
    :func:`run_scope` / :func:`compare_scopes`.

    ``res`` must carry ``circ1``, ``circ2``, ``topology``, ``angles``,
    ``target1`` and ``target2`` (``run_scope`` includes these). Saves a PNG when
    ``figure_save_name`` is given and returns the distributions dict.
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dists = post_optimization_distributions(
        res["circ1"], res["circ2"], res["topology"], res["angles"],
        res["target1"], res["target2"], nshots=nshots, backend=backend, norm=norm,
        basal1=res.get("basal1"), basal2=res.get("basal2"))

    from .plotting import state_occupancy
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    ngates = len(res.get("topology", []))
    fkl = res.get("final_kl", float("nan"))
    fig.suptitle(
        f"{title_prefix} — mono vs. QuantumXCT vs. co-culture "
        f"({ngates} gates, final KL {fkl:.4f}, {nshots} shots)", fontsize=14)
    for ax, mono, sim, tgt, name in [
        (axes[0], dists.get("mono1"), dists["sim1"], dists["tgt1"], "CT1"),
        (axes[1], dists.get("mono2"), dists["sim2"], dists["tgt2"], "CT2"),
    ]:
        nn = len(next(iter(sim))) if sim else 1
        state_occupancy(ax, mono, sim, tgt, nn, name, as_percent=False,
                        ylabel=(ax is axes[0]), title=name,
                        labels=("mono (basal)", "QuantumXCT (measured)", "co-culture (target)"))

    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    if figure_save_name:
        fig.savefig(figure_save_name, dpi=300)
        print(f"Saved histogram figure to {figure_save_name}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return dists


def plot_scope_report(
    res: Dict,
    gl_ct1: List[str],
    gl_ct2: List[str],
    valid_lr_pairs: List[Tuple[str, str]],
    out_dir: str = ".",
    scope: Optional[str] = None,
    nshots: Optional[int] = None,
    ct1_name: str = "Cell type 1",
    ct2_name: str = "Cell type 2",
    norm=None,
    show: bool = True,
    draw_circuit: bool = True,
):
    """Reproduce, for ONE scope result, the full figure set the functional
    notebook prints: the optimized circuit diagram, the measurement histograms,
    the 3-way (mono / QuantumXCT / co) occupancy bars, and the interaction
    network colored by cell type. Everything is shown inline and saved under
    ``out_dir``. Returns the ``contribution_df`` from the network analysis.

    ``res`` is a per-scope dict from :func:`run_scope`/:func:`compare_scopes`
    (carries circ1/circ2, topology, angles, targets, basal). ``gl_ct1``/``gl_ct2``
    are the gene names for each cell type (register); ``valid_lr_pairs`` the
    biologically allowed ligand->receptor pairs.
    """
    import os as _os
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _os.makedirs(out_dir, exist_ok=True)
    tag = scope or res.get("scope", "scope")
    nshots = nshots if nshots is not None else res.get("nshots", 5000)
    gene_list = list(gl_ct1) + list(gl_ct2)
    topology, angles = res["topology"], res["angles"]

    # 1) 3-way occupancy (mono / QuantumXCT / co), per register
    plot_scope_histograms(
        res, nshots=nshots, norm=norm, title_prefix=f"[{tag}]",
        figure_save_name=_os.path.join(out_dir, f"{tag}_occupancy_3way.png"), show=show)

    # Build the post-optimization measured circuit once (shared by 2 and 3)
    base = concatenate_circuits_with_separate_measurements(res["circ1"], res["circ2"])
    circ = add_crx_gates_and_measurements_to_circuit(
        base, res["circ1"].num_qubits, list(topology), list(angles))

    # 2) circuit diagram (text always; the colored compact figure is the standard,
    #    identical to make_search_figures — coloured state-prep boxes + CRX topology)
    if draw_circuit:
        print(f"\n[{tag}] optimized circuit ({len(topology)} CRX gates):")
        try:
            print(circ.draw("text"))
        except Exception as e:
            print(f"  (text draw unavailable: {e})")
        try:
            from .plotting import circuit_figure
            fig = circuit_figure(list(topology), list(angles), gene_list, len(gl_ct1))
            fig.savefig(_os.path.join(out_dir, f"{tag}_circuit_optimized.png"),
                        dpi=300, bbox_inches="tight")
            if show:
                plt.show()
            else:
                plt.close(fig)
        except Exception as e:
            print(f"  (circuit diagram skipped: {e})")

    # 3) measurement histograms of the optimized circuit
    plot_measurement_histograms(
        circ, nshots=nshots, title_prefix=f"[{tag}] optimized",
        figure_save_name=_os.path.join(out_dir, f"{tag}_measurement_histograms.png"))

    # 4) interaction network colored by cell type (needs the contribution table)
    contribution_df = None
    if len(topology) > 0:
        contribution_df, _hub, _genes = analyze_and_summarize_network(
            circ1=res["circ1"], circ2=res["circ2"], topology=topology, angles=angles,
            state_vec_probs_target1=res["target1"], state_vec_probs_target2=res["target2"],
            gene_list=gene_list, nshots=nshots)
        # LR-only ('inter') is unconstrained by design: show every DIRECT cross-cell
        # pair the search discovered, with no valid-LR bridge re-routing. Passing an
        # empty list makes the relay plot draw the raw cross edges as-is.
        is_inter = (scope or res.get("scope")) == "inter"
        lr_for_plot = [] if is_inter else valid_lr_pairs
        net_title = (f"[{tag}] Direct LR interaction network (unconstrained)"
                     if is_inter else f"[{tag}] Biologically Constrained Interaction Network")
        try:
            plot_quantum_relay_network(
                topology=topology, gene_list=gene_list, gl_ct1=gl_ct1, gl_ct2=gl_ct2,
                contribution_df=contribution_df, ct1_name=ct1_name, ct2_name=ct2_name,
                valid_lr_pairs=lr_for_plot,
                title=net_title,
                fname=_os.path.join(out_dir, f"{tag}_relay_network.png"))
        except Exception as e:
            print(f"  (relay network skipped: {e})")
    else:
        print(f"[{tag}] no gates found -> no interaction network to draw.")

    # 5) CRX angle-optimization trace (best-so-far), via the shared plotting function
    hist = res.get("angle_history")
    if hist:
        from .plotting import angle_trace
        figA, axA = plt.subplots(figsize=(6, 4.5))
        angle_trace(axA, hist, f"[{tag}] CRX Angle Optimization")
        figA.tight_layout()
        figA.savefig(_os.path.join(out_dir, f"{tag}_angle_opt.png"), dpi=300)
        if show:
            plt.show()
        else:
            plt.close(figA)

    return contribution_df


def _label(gene_list, control, target) -> str:
    if gene_list is None:
        return f"q{control} -> q{target}"
    return f"{gene_list[control]} -> {gene_list[target]}"


# ----------------------------------------------------------------------------
# Table 1 — cumulative ablation reduction (sequential)
# ----------------------------------------------------------------------------
def cumulative_ablation_table(
    circ1, circ2, topology, angles, target1, target2,
    gene_list: Optional[List[str]] = None, nshots: int = 5000, norm=None,
) -> pd.DataFrame:
    """Add gates back in order; record running KL and cumulative reduction.

    'Cumulative KL Reduction' is defined as (baseline_KL - current_KL), so it is
    positive when the added gates improve the fit.
    """
    base_kl = baseline_kl(circ1, circ2, target1, target2, nshots, norm)
    rows = [{
        "Step": 0, "Interaction": "Baseline (0 gates)", "CRX Angle (rad)": np.nan,
        "KL Divergence": base_kl, "KL Change": 0.0,
        "Cumulative KL Reduction": 0.0, "% of Baseline": 0.0,
    }]
    prev_kl = base_kl
    for i, ((c, t), ang) in enumerate(zip(topology, angles), start=1):
        kl = _kl_for_topology(circ1, circ2, topology[:i], angles[:i], target1, target2, nshots, norm)
        cum_reduction = base_kl - kl
        rows.append({
            "Step": i,
            "Interaction": _label(gene_list, c, t),
            "CRX Angle (rad)": ang,
            "KL Divergence": kl,
            "KL Change": kl - prev_kl,            # negative == improvement at this step
            "Cumulative KL Reduction": cum_reduction,
            "% of Baseline": (cum_reduction / base_kl * 100.0) if base_kl > 1e-12 else 0.0,
        })
        prev_kl = kl
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Table 2 — isolated + leave-one-out gate contribution (order-independent)
# ----------------------------------------------------------------------------
def isolated_contribution_table(
    circ1, circ2, topology, angles, target1, target2,
    gene_list: Optional[List[str]] = None, nshots: int = 5000, norm=None,
) -> pd.DataFrame:
    """For each gate: isolated effect (gate alone) and marginal leave-one-out effect.

    Columns:
      - 'Isolated KL'            : KL with only this single gate present.
      - 'Isolated Reduction'     : baseline_KL - Isolated KL  (this gate on its own).
      - 'LOO KL (without gate)'  : KL of the full circuit with this gate removed.
      - 'Marginal Contribution'  : LOO KL - full KL  (how much removing it hurts;
                                    positive == the gate is pulling its weight).
    """
    base_kl = baseline_kl(circ1, circ2, target1, target2, nshots, norm)
    full_kl = _kl_for_topology(circ1, circ2, topology, angles, target1, target2, nshots, norm)
    rows = []
    for i, ((c, t), ang) in enumerate(zip(topology, angles)):
        iso_kl = _kl_for_topology(circ1, circ2, [(c, t)], [ang], target1, target2, nshots, norm)
        loo_topo = list(topology[:i]) + list(topology[i + 1:])
        loo_ang = list(angles[:i]) + list(angles[i + 1:])
        loo_kl = _kl_for_topology(circ1, circ2, loo_topo, loo_ang, target1, target2, nshots, norm)
        rows.append({
            "Gate": _label(gene_list, c, t),
            "CRX Angle (rad)": ang,
            "Isolated KL": iso_kl,
            "Isolated Reduction": base_kl - iso_kl,
            "LOO KL (without gate)": loo_kl,
            "Marginal Contribution": loo_kl - full_kl,
            "% Marginal of Baseline": ((loo_kl - full_kl) / base_kl * 100.0) if base_kl > 1e-12 else 0.0,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Marginal Contribution", ascending=False).reset_index(drop=True)
    return df


# ----------------------------------------------------------------------------
# Per-scope run + cross-scope comparison
# ----------------------------------------------------------------------------
def run_scope(
    config: RunConfig,
    scope: str,
    counts_ct1_basal: dict, counts_ct2_basal: dict,
    counts_ct1_inter: dict, counts_ct2_inter: dict,
    gene_list: Optional[List[str]] = None,
) -> Dict:
    """Search + optimize for one scope; return topology, KLs and both tables."""
    import dataclasses
    cfg = dataclasses.replace(config, search=dataclasses.replace(config.search, candidate_scope=scope))
    qx = QuantumXct(cfg, gene_list=gene_list)
    qx.encode(counts_ct1_basal, counts_ct2_basal, counts_ct1_inter, counts_ct2_inter)

    # screen_candidates() is scope-aware: 'inter' => all cross-register pairs with
    # no density screening and no GRN; 'intra'/'all' => density-screened.
    qx.screen_candidates()
    if scope == "intra":
        qx.search_topology_grn_sequential()    # per-cell sequential GRN optimization
    else:
        qx.search_topology()
    qx.optimize_angles()

    # Evaluate at the SAME shot count the search used, so the reported KL is
    # consistent with what the topology was optimized against (mixing shot counts
    # makes tiny KL differences flip sign). Set config.backend.shots high enough
    # (>= 5000) for stable numbers.
    nshots = cfg.backend.shots
    norm = None  # amplitudes already encoded per cfg.encoding.norm
    base_kl = baseline_kl(qx.circ1, qx.circ2, qx.target1, qx.target2, nshots, norm)
    final_kl = _kl_for_topology(qx.circ1, qx.circ2, qx.result.topology, qx.result.angles,
                                qx.target1, qx.target2, nshots, norm)
    cum_df = cumulative_ablation_table(qx.circ1, qx.circ2, qx.result.topology, qx.result.angles,
                                       qx.target1, qx.target2, gene_list, nshots, norm)
    iso_df = isolated_contribution_table(qx.circ1, qx.circ2, qx.result.topology, qx.result.angles,
                                         qx.target1, qx.target2, gene_list, nshots, norm)
    return {
        "scope": scope,
        "topology": qx.result.topology,
        "angles": qx.result.angles,
        "baseline_kl": base_kl,
        "final_kl": final_kl,
        "reduction": base_kl - final_kl,
        "pct_reduction": (base_kl - final_kl) / base_kl * 100.0 if base_kl > 1e-12 else 0.0,
        "n_gates": len(qx.result.topology),
        "cumulative_ablation": cum_df,
        "isolated_contribution": iso_df,
        # Circuits + targets so callers can plot post-optimization histograms
        # (see plot_scope_histograms). In-memory objects, for notebook use.
        "circ1": qx.circ1,
        "circ2": qx.circ2,
        "target1": qx.target1,
        "target2": qx.target2,
        "basal1": qx.initial1,   # mono-culture initial-state amplitudes
        "basal2": qx.initial2,
        "nshots": nshots,
    }


def compare_scopes(
    config: RunConfig,
    counts_ct1_basal: dict, counts_ct2_basal: dict,
    counts_ct1_inter: dict, counts_ct2_inter: dict,
    gene_list: Optional[List[str]] = None,
    scopes: Tuple[str, ...] = ("all", "intra", "inter"),
) -> Dict:
    """Run every scope and return a summary DataFrame + the per-scope detail dicts."""
    scope_names = {"all": "GRN + LR", "intra": "GRN only (intra)", "inter": "LR only (inter)"}
    details = {}
    summary_rows = []
    for scope in scopes:
        res = run_scope(config, scope, counts_ct1_basal, counts_ct2_basal,
                        counts_ct1_inter, counts_ct2_inter, gene_list)
        details[scope] = res
        summary_rows.append({
            "Scope": scope_names.get(scope, scope),
            "Baseline KL": res["baseline_kl"],
            "Final KL": res["final_kl"],
            "KL Reduction": res["reduction"],
            "% Reduction": res["pct_reduction"],
            "# Gates": res["n_gates"],
        })
    return {"summary": pd.DataFrame(summary_rows), "detail": details}


# ----------------------------------------------------------------------------
# Benchmark: L1 vs L2 encoding (scaling) comparison — for the simulated cells
# ----------------------------------------------------------------------------
def compare_norms(
    config: RunConfig,
    counts_ct1_basal: dict, counts_ct2_basal: dict,
    counts_ct1_inter: dict, counts_ct2_inter: dict,
    gene_list: Optional[List[str]] = None,
    norms=("l1", "l2"),
    scope: str = "all",
    ground_truth_edges=None,
    include_recovery: bool = True,
    out_dir: Optional[str] = None,
    show: bool = True,
):
    """Compare L1 vs L2 amplitude encodings on the SAME data/pipeline.

    Set ``include_recovery=False`` for real datasets with no known ground-truth
    network (e.g. the ovarian data): the recovery columns are then omitted and
    only the KL comparison is reported.

    Scaling logic (see ``vector_normalize_dictionary_values``):
      * L2 (original paper, Eqs 1-2): amplitude = count / ||count||_2, so the
        measurement probability is proportional to count^2 — it *emphasizes common
        (high-count) states*.
      * L1 (QuantumXCT default): amplitude = sqrt(count / sum(count)), so the
        measurement probability equals the empirical frequency (count / total) —
        *frequency-preserving*.

    Because L1 and L2 define different target distributions, each norm's own KL is
    not directly comparable across norms. So for every norm we report:
      - own_final_kl / own_baseline_kl : the circuit vs its OWN encoded target
        (what the search optimized against);
      - ref_kl_vs_frequency : the optimized circuit's MEASURED output vs the true
        empirical co-culture frequency — a common yardstick both norms are scored on;
      - edge-recovery metrics (precision/recall/direction/AUC) vs the synthetic
        ground-truth network, which is encoding-independent.

    Returns {"summary": DataFrame(one row per norm), "detail": {...}}. Saves a CSV
    and a comparison figure to ``out_dir`` when given.
    """
    import dataclasses
    import time
    from .circuits import calculate_kl_divergence
    from .metrics import evaluate_edge_recovery

    def _freq(counts):
        tot = sum(counts.values())
        return {k: v / tot for k, v in counts.items()} if tot else {}

    def _decode_to_frequency(dist, norm):
        """Map a circuit's MEASURED Born probabilities back to frequency space, so
        the common-reference KL is fair across encodings.
          - L1: p_i = frequency already -> identity.
          - L2: p_i = count_i^2 / sum(count^2); frequency ∝ sqrt(p_i), renormalized.
        """
        if norm == "l2":
            import math
            s = {k: math.sqrt(max(v, 0.0)) for k, v in dist.items()}
            tot = sum(s.values())
            return {k: (v / tot if tot > 0 else 0.0) for k, v in s.items()}
        return dict(dist)

    ref1, ref2 = _freq(counts_ct1_inter), _freq(counts_ct2_inter)

    rows, detail = [], {}
    for norm in norms:
        cfg = dataclasses.replace(
            config,
            encoding=dataclasses.replace(config.encoding, norm=norm),
            search=dataclasses.replace(config.search, candidate_scope=scope),
        )
        qx = QuantumXct(cfg, gene_list=gene_list)
        qx.encode(counts_ct1_basal, counts_ct2_basal, counts_ct1_inter, counts_ct2_inter)
        qx.screen_candidates()
        t0 = time.perf_counter()
        if scope == "intra":
            qx.search_topology_grn_sequential()
        else:
            qx.search_topology()
        qx.optimize_angles()
        runtime = time.perf_counter() - t0

        nshots = cfg.backend.shots
        base = baseline_kl(qx.circ1, qx.circ2, qx.target1, qx.target2, nshots, None)
        final = _kl_for_topology(qx.circ1, qx.circ2, qx.result.topology, qx.result.angles,
                                 qx.target1, qx.target2, nshots, None)
        # common-reference KL: DECODE the measured output back to frequency space
        # (so L1 and L2 are compared fairly), then KL vs the true co-culture frequency.
        d = post_optimization_distributions(qx.circ1, qx.circ2, qx.result.topology,
                                             qx.result.angles, qx.target1, qx.target2,
                                             nshots=nshots, norm=None)
        dec1 = _decode_to_frequency(d["sim1"], norm)
        dec2 = _decode_to_frequency(d["sim2"], norm)
        ref_kl = calculate_kl_divergence(dec1, ref1) + calculate_kl_divergence(dec2, ref2)

        row = {
            "norm": norm, "scope": scope, "n_gates": len(qx.result.topology),
            "own_baseline_kl": round(float(base), 4), "own_final_kl": round(float(final), 4),
            "own_pct_reduction": round((base - final) / base * 100 if base > 1e-12 else 0.0, 2),
            "ref_kl_vs_frequency": round(float(ref_kl), 4),
            "runtime_s": round(float(runtime), 2),
        }
        rec = None
        # recovery needs a known ground-truth network; auto-skip when absent so the
        # function safely reduces to just the L1-vs-L2 KL comparison (e.g. ovarian).
        if include_recovery and ground_truth_edges is not None:
            cdf = qx.analyze() if gene_list is not None else None
            rec = evaluate_edge_recovery(qx.result.topology, gene_list,
                                         ground_truth_edges=ground_truth_edges,
                                         contribution_df=cdf, show=False)
            s = rec["summary"].iloc[0]
            row.update({"precision": s["precision"], "recall": s["recall"],
                        "direction_accuracy": s["direction_accuracy"], "roc_auc": s["roc_auc"]})
        rows.append(row)
        detail[norm] = {"recovery": rec, "topology": qx.result.topology, "distributions": d}

    summary = pd.DataFrame(rows)
    if out_dir:
        import os as _os
        _os.makedirs(out_dir, exist_ok=True)
        summary.to_csv(_os.path.join(out_dir, "norm_comparison_summary.csv"), index=False)
        _plot_norm_comparison(summary, out_dir, show)
    return {"summary": summary, "detail": detail}


def _plot_norm_comparison(summary, out_dir, show):
    import os as _os
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as _np

    norms = list(summary["norm"])
    x = _np.arange(len(norms))
    has_recovery = "precision" in summary.columns
    npanels = 3 if has_recovery else 2
    fig, axes = plt.subplots(1, npanels, figsize=(5 * npanels, 4.5))
    # (a) KL: own vs common-reference
    ax = axes[0]; w2 = 0.35
    ax.bar(x - w2 / 2, summary["own_final_kl"], w2, label="own-encoding KL", color="#56B4E9", alpha=0.85)
    ax.bar(x + w2 / 2, summary["ref_kl_vs_frequency"], w2, label="KL vs true frequency", color="#E69F00", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(norms); ax.set_title("KL by encoding"); ax.legend(frameon=False)
    # (b) recovery metrics (only when a ground truth was scored)
    if has_recovery:
        ax = axes[1]
        mets = ["precision", "recall", "direction_accuracy", "roc_auc"]
        ww = 0.8 / len(norms)
        for i, nm in enumerate(norms):
            row = summary[summary["norm"] == nm].iloc[0]
            vals = [row[m] if row[m] == row[m] else 0.0 for m in mets]
            ax.bar(_np.arange(len(mets)) + i * ww, vals, ww, label=nm)
        ax.set_xticks(_np.arange(len(mets)) + ww * (len(norms) - 1) / 2)
        ax.set_xticklabels(mets, rotation=25, ha="right"); ax.set_ylim(0, 1.05)
        ax.set_title("edge recovery vs ground truth"); ax.legend(frameon=False)
    # (c) gates + runtime
    ax = axes[-1]
    ax.bar(x - 0.2, summary["n_gates"], 0.4, label="# gates", color="#56B4E9", alpha=0.85)
    ax2 = ax.twinx(); ax2.bar(x + 0.2, summary["runtime_s"], 0.4, label="runtime (s)", color="#E69F00", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(norms); ax.set_ylabel("# gates"); ax2.set_ylabel("runtime (s)")
    ax.set_title("cost")
    fig.suptitle("L1 vs L2 encoding comparison", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(_os.path.join(out_dir, "norm_comparison.png"), dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)
