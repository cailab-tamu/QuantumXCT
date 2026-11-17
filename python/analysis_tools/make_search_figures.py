"""Search-result figures for the paper/poster — rendered through qxct.plotting
(the single source of truth), scored by the REAL Aer shot pipeline by default.

For each dataset (synthetic, ovarian) x encoding (L1, L2) x algorithm
(iterative_n_wise, multi_epoch, vqe_hybrid) x scope (GRN=intra, LR=inter,
GRN+LR=all) it runs the object-oriented QuantumXCT pipeline, then draws:

  A  data joint histograms (gauge probability), non-interacting & interacting
  per algorithm x scope:
     circuit, interaction network, state-occupancy comparison (CT1, CT2),
     CRX angle-optimization trace
  summary: % KL reduction per algorithm x scope, and the three GRN+LR traces.

Scoring mode:
  DEFAULT = the real Aer shot pipeline (--shots, default 5000).  The search,
            the angle optimization, and the occupancy "QXCT" bars are all
            sampled — the same pipeline the manuscript describes.
  --exact = shot-noise-free statevector scoring, for the SI verification only.

Arrow width in the network = the gate's leave-one-out KL contribution, computed
with exact probabilities (as the manuscript's contribution analysis is).

Run from QuantumXCT/python:
    python analysis_tools/make_search_figures.py                      # all, sampled
    python analysis_tools/make_search_figures.py --datasets ovarian --norms l1
    python analysis_tools/make_search_figures.py --exact              # SI verification
Outputs (sampled and exact go to SEPARATE folders so they never overwrite):
    results/analyses/search_panels/{ds}_{norm}_{sampled|exact}/*.png|pdf|svg
    results/analyses/search_figures/{ds}_{norm}_{sampled|exact}_summary.csv (+ _traces.json)
"""
import argparse, io, contextlib, json, os, time
import sys
sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import qxct.circuits as _circ
import qxct.analysis as _ana
_SAMPLED_SCORER = _circ.score_circuit_kl_divergences        # pristine Aer scorer (capture BEFORE any patch)

from analysis_tools.run_search_exact import cfg, lab           # same configs as the SI tables (import patches exact)
from analysis_tools.data import load_ovarian, load_sim, OV_CT1, OV_CT2, SIM_CT1, SIM_CT2
from qxct import QuantumXct
from qxct.exact import ExactScorer, marginals, use_exact_scoring
from qxct.circuits import optimize_crx_angles
from qxct.analysis import post_optimization_distributions
from LR_databases.lr_pairs import curate_lr_pairs
import qxct.plotting as pl


def use_sampled_scoring():
    """Restore the real Aer shot-based scorer (undo any exact-scoring patch)."""
    _circ.score_circuit_kl_divergences = _SAMPLED_SCORER
    _ana.score_circuit_kl_divergences = _SAMPLED_SCORER


ALGOS = [("iterative_n_wise", "N-wise"), ("multi_epoch", "Multi-Epoch"), ("vqe_hybrid", "VQE + Multi-Epoch")]
SCOPES = [("intra", "GRN"), ("inter", "LR"), ("all", "GRN + LR")]
DATASETS = {
    "synthetic": dict(loader=load_sim, ct1=SIM_CT1, ct2=SIM_CT2, name1="Cell type A", name2="Cell type B", kl_tol=0.05),
    "ovarian": dict(loader=load_ovarian, ct1=OV_CT1, ct2=OV_CT2, name1="Cancer cells", name2="Fibroblasts (CAFs)", kl_tol=0.01),
}
plt.rcParams.update({"font.size": 11, "font.weight": "bold", "axes.titleweight": "bold",
                     "axes.labelweight": "bold", "axes.titlesize": 13})


def save(fig, path):
    fig.savefig(path + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(path + ".pdf", bbox_inches="tight")
    fig.savefig(path + ".svg", bbox_inches="tight")
    plt.close(fig)


def top_k_candidates(sc, cands, k):
    """Rank candidates by exact single-gate KL reduction (for the VQE QUBO size cap)."""
    if len(cands) <= k:
        return list(cands)
    b = sc.baseline()
    gain = [(b - sc.optimize_angles([c], restarts=2)[1], c) for c in cands]
    return [c for _, c in sorted(gain, key=lambda x: -x[0])[:k]]


def run_one(H, genes, norm, scope, algo, kl_tol, vqe_k, shots=5000):
    """Run the OO pipeline once. Scoring mode (sampled vs exact) is whatever the
    global scorer is set to by main()."""
    c = cfg(norm, scope=scope, kl_tol=kl_tol, algorithm=algo)
    c.backend.shots = shots
    if algo == "vqe_hybrid":
        c.search.vqe_niter = 150
    qx = QuantumXct(c, gene_list=genes)
    qx.encode(H["ct1_mo"], H["ct2_mo"], H["ct1_co"], H["ct2_co"])
    sc = ExactScorer(qx.initial1, qx.initial2, qx.target1, qx.target2)   # exact reference for % and LOO widths
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        cands = qx.screen_candidates()
        if algo == "vqe_hybrid":
            cands = top_k_candidates(sc, cands, vqe_k)
            qx.result.candidates = cands
        if scope == "intra" and algo == "multi_epoch":
            qx.search_topology_grn_sequential()
        else:
            qx.search_topology(cands)
        topo = list(qx.result.topology or [])
        o = c.optimizer
        if topo:
            angles, _kl, hist = optimize_crx_angles(qx.circ1, qx.circ2, qx.target1, qx.target2,
                                                    cnot_topology=topo, nshots=c.backend.shots, etol=o.etol,
                                                    opt_method=o.method, initial_angle_value=o.initial_angle)
        else:
            angles, hist = [], []
    angles = [float(a) for a in angles]
    base, final = sc.baseline(), sc.kl(topo, angles)
    return dict(qx=qx, sc=sc, topo=topo, angles=angles, hist=[float(h) for h in (hist or [])],
                n_cands=len(cands), base=base, final=final, pct=100 * (base - final) / base, sec=time.time() - t0)


def loo_contributions(sc, topo, angles):
    """Exact leave-one-out KL contribution of each gate (arrow width)."""
    full = sc.kl(topo, angles)
    return [max(sc.kl(topo[:i] + topo[i + 1:], angles[:i] + angles[i + 1:]) - full, 0.0)
            for i in range(len(topo))]

ap = argparse.ArgumentParser()
ap.add_argument("--datasets", nargs="+", default=["synthetic", "ovarian"])
ap.add_argument("--norms", nargs="+", default=["l1", "l2"])
ap.add_argument("--vqe-max-cands", type=int, default=16)
ap.add_argument("--algos", nargs="+", default=[a for a, _ in ALGOS])
ap.add_argument("--exact", action="store_true",
                help="exact statevector scoring (SI verification). Default: real Aer shot sampling.")
ap.add_argument("--shots", type=int, default=5000, help="shots for the sampled pipeline (default 5000)")
a = ap.parse_args()
sampled = not a.exact
if a.exact:
    use_exact_scoring(True)
else:
    use_sampled_scoring()
mode = f"{a.shots}-shot Aer sampling" if sampled else "exact statevector"
suffix = "sampled" if sampled else "exact"     # keeps sampled and exact outputs in separate folders
print(f"[make_search_figures] scoring mode: {mode}  ->  *_{suffix} folders", flush=True)

for ds in a.datasets:
    D = DATASETS[ds]; H = D["loader"](); genes = D["ct1"] + D["ct2"]; n1 = len(D["ct1"]); n2 = len(D["ct2"])
    lr_pairs = curate_lr_pairs(genes, verbose=False)   # validated ligand-receptor links within the panel (plot marking only)
    for norm in a.norms:
        out = f"results/analyses/search_panels/{ds}_{norm}_{suffix}"; os.makedirs(out, exist_ok=True)
        res_dir = "results/analyses/search_figures"; os.makedirs(res_dir, exist_ok=True)
        rows, traces = [], {}

        # A: data joint histograms in the encoding gauge (exact encoded data, not a model output)
        qx0 = QuantumXct(cfg(norm), gene_list=genes)
        qx0.encode(H["ct1_mo"], H["ct2_mo"], H["ct1_co"], H["ct2_co"])
        sc0 = ExactScorer(qx0.initial1, qx0.initial2, qx0.target1, qx0.target2)
        p01, p02 = marginals(sc0.psi0, sc0.n1, sc0.n2)
        fig, axs = plt.subplots(2, 2, figsize=(10, 6.5))
        for ax, vec, nn, ttl in [(axs[0, 0], p01, n1, "Data Joint Histogram Non-interacting (CT1)"),
                                 (axs[0, 1], p02, n2, "Data Joint Histogram Non-interacting (CT2)"),
                                 (axs[1, 0], sc0.q1, n1, "Data Joint Histogram Interacting (CT1)"),
                                 (axs[1, 1], sc0.q2, n2, "Data Joint Histogram Interacting (CT2)")]:
            col = pl.C_INT if "Interacting" in ttl and "Non" not in ttl else pl.C_NON
            ax.bar(range(2 ** nn), vec, color=col)
            ax.set_xticks(range(2 ** nn)); ax.set_xticklabels(pl.bitlabels(nn), rotation=90, fontsize=7 if nn > 3 else 9)
            ax.set_title(ttl, fontsize=10); ax.set_ylabel("Probability"); ax.set_xlabel("State (bit 1 = rightmost gene)")
        fig.suptitle(f"{ds} · {norm.upper()} gauge · CT1 = {D['name1']} ({', '.join(D['ct1'])}), "
                     f"CT2 = {D['name2']} ({', '.join(D['ct2'])})", fontsize=9)
        fig.tight_layout(); save(fig, f"{out}/A_data_histograms")

        for algo, aname in ALGOS:
            if algo not in a.algos:
                continue
            for scope, sname in SCOPES:
                tag = f"{algo}_{scope}"
                try:
                    r = run_one(H, genes, norm, scope, algo, D["kl_tol"], a.vqe_max_cands, shots=a.shots)
                except Exception as e:
                    rows.append(dict(algorithm=algo, scope=scope, error=repr(e)[:300]))
                    print(ds, norm, tag, "FAILED", e, flush=True); continue
                topo, ang, sc, qx = r["topo"], r["angles"], r["sc"], r["qx"]
                traces[tag] = r["hist"]

                # circuit
                if topo:
                    save(pl.circuit_figure(topo, ang, genes, n1), f"{out}/{tag}_circuit")

                # angle-optimization trace (shot-based when sampled)
                fig, axt = plt.subplots(figsize=(4.6, 3.8))
                pl.angle_trace(axt, r["hist"], f"CRX Angle Optimization {aname} "
                               f"({a.shots} shots)" if sampled else f"CRX Angle Optimization {aname} (exact)")
                fig.tight_layout(); save(fig, f"{out}/{tag}_angle_opt")

                # state-occupancy: model bars from the REAL Aer measurement (or exact if --exact)
                if topo:
                    if sampled:
                        d = post_optimization_distributions(qx.circ1, qx.circ2, topo, ang, qx.target1, qx.target2,
                                                            nshots=a.shots, norm=None,
                                                            basal1=qx.initial1, basal2=qx.initial2)
                        m1, pm1, t1 = d["mono1"], d["sim1"], d["tgt1"]
                        m2, pm2, t2 = d["mono2"], d["sim2"], d["tgt2"]
                    else:
                        e1, e2 = marginals(sc.state(topo, ang), sc.n1, sc.n2)
                        m1, pm1, t1 = p01, e1, sc.q1
                        m2, pm2, t2 = p02, e2, sc.q2
                else:
                    m1, pm1, t1 = p01, p01, sc.q1
                    m2, pm2, t2 = p02, p02, sc.q2
                fig, (o1, o2) = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True,
                                             gridspec_kw={"width_ratios": [1, 1.25]})
                pl.state_occupancy(o1, m1, pm1, t1, n1, "CT1")
                pl.state_occupancy(o2, m2, pm2, t2, n2, "CT2", ylabel=False)
                fig.suptitle(f"QXCT = {a.shots}-shot Aer sampling" if sampled else "QXCT = exact statevector",
                             fontsize=9, y=1.02)
                fig.tight_layout(); save(fig, f"{out}/{tag}_occupancy")

                # interaction network (exact leave-one-out widths, as the paper's contributions);
                # validated ligand-receptor links are marked via the LR database (lr_pairs).
                loo = loo_contributions(sc, topo, ang)
                fig, axn = plt.subplots(figsize=(6, 3.9))
                pl.interaction_network(axn, topo, genes, n1, contributions=loo,
                                       name1=D["name1"], name2=D["name2"], bidirectional=True,
                                       lr_pairs=lr_pairs)
                axn.set_title(f"{aname} · {sname} · {len(topo)} gates · {r['pct']:.1f}% KL reduction",
                              fontsize=10, pad=22)
                save(fig, f"{out}/{tag}_network")

                # restricted twin for the full GRN+LR network: validated LR hubs only
                if scope == "all" and lr_pairs and topo:
                    fig, axr = plt.subplots(figsize=(6, 3.9))
                    pl.interaction_network(axr, topo, genes, n1, contributions=loo,
                                           name1=D["name1"], name2=D["name2"], bidirectional=True,
                                           lr_pairs=lr_pairs, lr_only=True)
                    axr.set_title(f"{aname} · {sname} (LR-restricted) · {r['pct']:.1f}% KL reduction",
                                  fontsize=10, pad=22)
                    save(fig, f"{out}/{tag}_network_lr_restricted")

                rows.append(dict(algorithm=algo, scope=scope, n_candidates=r["n_cands"], n_gates=len(topo),
                                 edges=";".join(lab(genes, topo)), angles=";".join(f"{x:.4f}" for x in ang),
                                 loo_kl=";".join(f"{x:.4f}" for x in loo),
                                 base=r["base"], final=r["final"], pct=r["pct"], scoring=mode, sec=r["sec"]))
                print(ds, norm, tag, f"{len(topo)} gates", f"{r['pct']:.1f}%", f"{r['sec']:.0f}s", flush=True)

        df = pd.DataFrame(rows); df.to_csv(f"{res_dir}/{ds}_{norm}_{suffix}_summary.csv", index=False)
        json.dump(traces, open(f"{res_dir}/{ds}_{norm}_{suffix}_traces.json", "w"))

        # summary: % reduction per algorithm x scope
        okd = df.dropna(subset=["pct"]) if "pct" in df else df
        fig, ax = plt.subplots(figsize=(7, 3.6)); w = 0.26
        for k, (algo, aname) in enumerate(ALGOS):
            vals = [okd[(okd.algorithm == algo) & (okd.scope == s)].pct.max()
                    if len(okd[(okd.algorithm == algo) & (okd.scope == s)]) else np.nan for s, _ in SCOPES]
            bars = ax.bar(np.arange(3) + (k - 1) * w, vals, w, label=aname,
                          color=["tab:blue", pl.C_QX, pl.C_INT][k])
            ax.bar_label(bars, fmt="%.0f", fontsize=8, padding=2)
        ax.set_xticks(range(3)); ax.set_xticklabels([s for _, s in SCOPES]); ax.set_ylabel("% KL reduction")
        ax.set_ylim(0, 110); ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
        ax.set_title(f"Search algorithms × scope ({ds}, {norm.upper()}, {mode})", fontsize=10)
        fig.tight_layout(); save(fig, f"{out}/summary_scope_algorithm")

        # summary: GRN+LR angle traces overlaid
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        for k, (algo, aname) in enumerate(ALGOS):
            if f"{algo}_all" in traces:
                pl.angle_trace(ax, traces[f"{algo}_all"], f"CRX Angle Optimization, GRN + LR ({ds}, {norm.upper()})",
                               color=["tab:blue", pl.C_QX, pl.C_INT][k], label=aname)
        ax.legend(frameon=False, fontsize=8); fig.tight_layout(); save(fig, f"{out}/summary_angle_traces_GRN_LR")
