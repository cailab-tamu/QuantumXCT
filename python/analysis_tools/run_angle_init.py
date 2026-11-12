"""Angle-initialization study (0 vs pi/2) for BOTH datasets.

The ovarian notebook already produces *_angle_init_curves.png; the simulated
notebook did not.  This reproduces the study for the synthetic AND the ovarian
data so both are regenerable and appear side by side.

For each dataset it takes the GRN+LR (all-scope) topology found by the
multi-epoch search under L1, then re-optimizes its CRX angles twice, starting
from 0 and from pi/2, and records the convergence of the combined KL.  This is
the local-minima question from the manuscript: does the identity start (0) get
stuck, or do both starts reach the same optimum?

Scoring: DEFAULT = the real Aer shot pipeline (--shots, default 5000), matching
the notebook.  --exact uses statevector scoring (shot-noise-free) for the SI.

Outputs (results/analyses):
    angle_init_study.csv                        dataset, init, final_kl, n_iters, angles
    figures/{synthetic,ovarian}_angle_init_curves.png/.pdf/.svg
Also copies each PNG into results/{classic_sim,ovarian_cocultured}/.
"""
import os, sys, io, contextlib, shutil, argparse
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import qxct.circuits as _circ
import qxct.analysis as _ana
_SAMPLED_SCORER = _circ.score_circuit_kl_divergences        # pristine Aer scorer (before any patch)

from qxct import QuantumXct, RunConfig, EncodingConfig, BackendConfig, SearchConfig, OptimizerConfig
from qxct.exact import use_exact_scoring, ExactScorer
from qxct.circuits import compare_angle_initializations
import qxct.plotting as pl
from analysis_tools.data import load_ovarian, load_sim, OV_CT1, OV_CT2, SIM_CT1, SIM_CT2


def use_sampled_scoring():
    _circ.score_circuit_kl_divergences = _SAMPLED_SCORER
    _ana.score_circuit_kl_divergences = _SAMPLED_SCORER


OUT = "results/analyses"
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

DATASETS = [
    ("synthetic", load_sim, SIM_CT1, SIM_CT2, 0.05, "results/classic_sim"),
    ("ovarian",  load_ovarian, OV_CT1, OV_CT2, 0.01, "results/ovarian_cocultured"),
]

ap = argparse.ArgumentParser()
ap.add_argument("--exact", action="store_true", help="exact statevector scoring (SI). Default: Aer sampling.")
ap.add_argument("--shots", type=int, default=5000)
args = ap.parse_args()
if args.exact:
    use_exact_scoring(True)
else:
    use_sampled_scoring()
mode = f"{args.shots}-shot Aer sampling" if not args.exact else "exact statevector"
suffix = "exact" if args.exact else "sampled"    # separate files for exact vs sampled
print(f"[run_angle_init] scoring mode: {mode}  ->  *_{suffix}", flush=True)
rows = []
for name, loader, ct1, ct2, kl_tol, mirror_dir in DATASETS:
    H = loader(); genes = ct1 + ct2
    c = RunConfig(encoding=EncodingConfig(norm="l1"), backend=BackendConfig(shots=args.shots),
                  search=SearchConfig(algorithm="multi_epoch", threshold=0.005, search_mode="pos",
                                      kl_tol=kl_tol, max_greedy_depth=30, candidate_scope="all"),
                  optimizer=OptimizerConfig(method="COBYLA", initial_angle=np.pi / 2))
    qx = QuantumXct(c, gene_list=genes)
    qx.encode(H["ct1_mo"], H["ct2_mo"], H["ct1_co"], H["ct2_co"])
    with contextlib.redirect_stdout(io.StringIO()):
        qx.screen_candidates(); qx.search_topology()
    topo = list(qx.result.topology)
    png = os.path.join(FIG, f"{name}_angle_init_curves_{suffix}.png")
    with contextlib.redirect_stdout(io.StringIO()):
        curves = compare_angle_initializations(
            qx.circ1, qx.circ2, qx.target1, qx.target2, topo,
            inits=(0.0, np.pi / 2), nshots=args.shots,
            opt_method="COBYLA", show_plot=False, filename=png)
    # matching PDF
    if os.path.exists(png):
        plt.close("all")
    for init, r in curves.items():
        rows.append(dict(dataset=name, init=round(float(init), 4),
                         final_kl=float(r["final_kl"]), n_iters=len(r["history"]),
                         n_gates=len(topo),
                         angles=";".join(f"{a:.4f}" for a in r["angles"])))
    # redraw as a clean labelled figure (png + pdf) and mirror into the dataset folder
    fig, ax = plt.subplots(figsize=(6, 4.5))
    lab = {0.0: r"start $\theta_0=0$ (identity)", float(np.pi / 2): r"start $\theta_0=\pi/2$"}
    for init, r in curves.items():
        h = r["history"]
        if h:
            hb = np.asarray(h, float)   # RAW per-evaluation objective (shows COBYLA exploring/escaping)
            ax.plot(range(len(hb)), hb, "-o", ms=3, lw=1.2,
                    label=f"{lab.get(float(init), init)}  (final KL {r['final_kl']:.4f})")
    ax.set_xlabel("Optimization Iteration"); ax.set_ylabel("Combined KL Divergence")
    ax.set_title(f"CRX angle-initialization study ({name}, GRN + LR, {len(topo)} gates)")
    ax.grid(True, ls=":", alpha=0.6); ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(png, dpi=300); fig.savefig(png[:-4] + ".pdf"); fig.savefig(png[:-4] + ".svg")
    plt.close(fig)
    if os.path.isdir(mirror_dir):
        shutil.copy(png, os.path.join(mirror_dir, f"{name}_angle_init_curves_{suffix}.png"))
    finals = {round(float(k), 4): v["final_kl"] for k, v in curves.items()}
    print(f"{name}: topo {len(topo)} gates; final KL  0-init={finals.get(0.0):.5f}  "
          f"pi/2-init={finals.get(round(float(np.pi/2),4)):.5f}", flush=True)

csv = os.path.join(OUT, f"angle_init_study_{suffix}.csv")
pd.DataFrame(rows).to_csv(csv, index=False)
print("wrote", csv)
