"""Ligand-receptor interaction networks: restricted vs unrestricted views, at a
best-effort and two interpretable search tolerances.

The QuantumXCT topology search is NEVER restricted here: it always discovers the
full best-effort network over all candidate links. The ligand-receptor database
is used only at drawing time, to decide which discovered cross-cell hubs are
validated biology:

  * unrestricted = the full discovered network. Validated ligand-receptor links
                   are drawn solid; other cross-cell links dashed.
  * restricted   = the same network with the unvalidated cross-cell links hidden,
                   leaving only the validated ligand-receptor hubs (and the
                   within-cell links).

kl_tol is the selection tolerance from the search (the minimum KL a longer
network must save to be preferred, circuits.py "Occam's Razor"). Smaller keeps
more links (best effort); larger keeps fewer, stronger links (interpretable):

  * best effort   : kl_tol = 0.05   (low KL, link-rich)
  * interpretable : kl_tol = 0.1 and 0.2 (fewer, stronger links)

Scoring is exact (shot-noise-free); arrow widths are the exact leave-one-out KL
contribution of each link. Every run is also saved to one JSON so the figures
can be regenerated or restyled without re-running the search.

Run from QuantumXCT/python:
    python analysis_tools/make_lr_networks.py                 # ovarian
    python analysis_tools/make_lr_networks.py ovarian synthetic
Outputs:
    results/analyses/lr_networks/{ds}_{tol}_{restricted|unrestricted}.{png,pdf,svg}
    results/analyses/lr_networks/{ds}_lr_networks.json
"""
import os, sys, io, json, contextlib
sys.path.insert(0, os.getcwd())
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qxct import QuantumXct
from qxct.exact import use_exact_scoring, ExactScorer
from LR_databases.lr_pairs import curate_lr_pairs
import qxct.plotting as pl
from analysis_tools.run_search_exact import cfg, lab
from analysis_tools.data import load_ovarian, load_sim, OV_CT1, OV_CT2, SIM_CT1, SIM_CT2

# scoring mode (sampled default / --exact opt-in) is decided in the arg block below

# --- figure style knobs (match the poster network look) ---------------------
# Bigger canvas -> bigger circles -> the gene labels fit at a readable size.
# The gray legend wraps to two lines automatically (handled in interaction_network).
FIGSIZE = (12, 7.4)
LABEL_FS = 18      # gene-name node labels
NAME_FS = 22       # cell-type column titles
CAPTION_FS = 14    # gray legend line
TITLE_FS = 16      # panel title

# (label, kl_tol): best effort = most links / lowest KL; larger tol = fewer links.
# best effort 0.01 keeps both validated hubs (PDGF + TGF-beta axis); 0.1 keeps the
# single strongest PDGF hub.
TOLS = [("best_effort", 0.01), ("balanced", 0.05), ("interpretable", 0.1)]
DATASETS = {
    "ovarian": dict(loader=load_ovarian, ct1=OV_CT1, ct2=OV_CT2,
                    name1="Cancer cells", name2="Fibroblasts (CAFs)"),
    "synthetic": dict(loader=load_sim, ct1=SIM_CT1, ct2=SIM_CT2,
                      name1="Cell type A", name2="Cell type B"),
}
plt.rcParams.update({"font.size": 11, "font.weight": "bold", "axes.titleweight": "bold"})


def loo_contributions(sc, topo, angles):
    """Exact leave-one-out KL contribution of each link (sets the arrow width)."""
    full = sc.kl(topo, angles)
    return [max(sc.kl(topo[:i] + topo[i + 1:], angles[:i] + angles[i + 1:]) - full, 0.0)
            for i in range(len(topo))]


def run_topo(H, genes, kl_tol):
    """Discover the best-effort topology at this selection tolerance (exact scoring)."""
    c = cfg(NORM, kl_tol=kl_tol, scope="all", algorithm="multi_epoch")
    qx = QuantumXct(c, gene_list=genes)
    qx.encode(H["ct1_mo"], H["ct2_mo"], H["ct1_co"], H["ct2_co"])
    with contextlib.redirect_stdout(io.StringIO()):
        qx.screen_candidates()
        qx.search_topology()
        qx.optimize_angles()
    _sc_exact = ExactScorer(qx.initial1, qx.initial2, qx.target1, qx.target2)
    # edge widths (LOO) + %: sampled 5000-shot (seeded) by default, exact only with --exact
    sc = _sc_exact if EXACT else _sc_exact.qiskit(mode="sampler", shots=5000, seed=42)
    topo = [tuple(e) for e in (qx.result.topology or [])]
    ang = [float(x) for x in (qx.result.angles or [])]
    return sc, topo, ang


def save(fig, path):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(path + "." + ext, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_label(ds, D, genes, n1, lr_pairs, label, tol, topo, loo, pct):
    """Draw the unrestricted / restricted / redirected views for one run, at the
    style knobs above. Used by both the full run and --plot-only. Returns the
    redirected-hub record (or None)."""
    for restricted in (False, True):
        if restricted and not lr_pairs:
            continue                                   # nothing to restrict to (e.g. synthetic)
        fig, ax = plt.subplots(figsize=FIGSIZE)
        pl.interaction_network(ax, topo, genes, n1, contributions=loo,
                               name1=D["name1"], name2=D["name2"], bidirectional=True,
                               lr_pairs=lr_pairs, lr_only=restricted,
                               label_fontsize=LABEL_FS, name_fontsize=NAME_FS,
                               caption_fontsize=CAPTION_FS)
        kind = "restricted" if restricted else "unrestricted"
        ax.set_title(f"{ds} · {kind} · kl_tol={tol} · {len(topo)} gates · "
                     f"{pct:.1f}% KL reduction", fontsize=TITLE_FS, pad=22)
        save(fig, f"{OUT}/{ds}_{label}_{kind}{TAG}")
    # redirected-hub view: non-LR cross-cell links routed through the ligand hub
    if not lr_pairs:
        return None
    new, kinds, neww = pl.redirect_through_hub(topo, genes, n1, lr_pairs, contributions=loo)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    pl.interaction_network(ax, new, genes, n1, contributions=neww,
                           name1=D["name1"], name2=D["name2"], bidirectional=True,
                           lr_pairs=lr_pairs,
                           label_fontsize=LABEL_FS, name_fontsize=NAME_FS,
                           caption_fontsize=CAPTION_FS)
    ax.set_title(f"{ds} · redirected-hub · kl_tol={tol} · {pct:.1f}% KL reduction",
                 fontsize=TITLE_FS, pad=22)
    save(fig, f"{OUT}/{ds}_{label}_redirected{TAG}")
    return {"topology": [list(e) for e in new], "edges": lab(genes, new), "kinds": kinds}


args = sys.argv[1:]
PLOT_ONLY = "--plot-only" in args
# --norm l1|l2 (default l1). Main Figs 2-3 use l2; l2 outputs get an _l2 suffix so they never clobber l1.
NORM = "l1"
if "--norm" in args:
    _i = args.index("--norm")
    if _i + 1 < len(args):
        NORM = args[_i + 1]
        args = args[:_i] + args[_i + 2:]
EXACT = "--exact" in args
if EXACT:
    args = [a for a in args if a != "--exact"]
    use_exact_scoring(True)   # SI-verification mode: exact search + exact edge widths
# default (no --exact): the real Aer 5000-shot sampled pipeline, seeded -> reproducible
TAG = ("" if NORM == "l1" else f"_{NORM}") + ("_exact" if EXACT else "")
datasets = [a for a in args if not a.startswith("-")] or ["ovarian"]
OUT = "results/analyses/lr_networks"
os.makedirs(OUT, exist_ok=True)

for ds in datasets:
    D = DATASETS[ds]

    if PLOT_ONLY:
        # restyle the saved networks with NO search: read the topology + LOO widths
        # from the JSON and re-draw at the current style knobs.
        with open(f"{OUT}/{ds}_lr_networks{TAG}.json") as f:
            rec = json.load(f)
        genes, n1 = rec["genes"], rec["n1"]
        lr_pairs = [tuple(p) for p in rec["lr_pairs"]]
        for label, run in rec["runs"].items():
            plot_label(ds, D, genes, n1, lr_pairs, label, run["kl_tol"],
                       [tuple(e) for e in run["topology"]], run["loo"], run["pct"])
            print(f"[plot-only] {ds} {label}: {run['n_gates']} gates -> figures restyled", flush=True)
        continue

    H = D["loader"]()
    genes = D["ct1"] + D["ct2"]
    n1 = len(D["ct1"])
    lr_pairs = curate_lr_pairs(genes)                      # validated LR within the panel
    record = {"dataset": ds, "genes": genes, "n1": n1,
              "lr_pairs": [list(p) for p in lr_pairs], "runs": {}}

    for label, tol in TOLS:
        sc, topo, ang = run_topo(H, genes, tol)
        loo = loo_contributions(sc, topo, ang)
        base, final = sc.baseline(), sc.kl(topo, ang)
        pct = 100.0 * (base - final) / base if base else 0.0
        record["runs"][label] = {
            "kl_tol": tol, "topology": [list(e) for e in topo], "edges": lab(genes, topo),
            "angles": ang, "loo": loo, "base": base, "final": final,
            "pct": pct, "n_gates": len(topo)}
        redirected = plot_label(ds, D, genes, n1, lr_pairs, label, tol, topo, loo, pct)
        if redirected is not None:
            record["runs"][label]["redirected"] = redirected
        print(f"{ds} {label} (kl_tol={tol}): {len(topo)} gates, {pct:.1f}% KL reduction; "
              f"edges={lab(genes, topo)}", flush=True)

    json.dump(record, open(f"{OUT}/{ds}_lr_networks{TAG}.json", "w"), indent=2)
    print(f"wrote {OUT}/{ds}_lr_networks{TAG}.json", flush=True)
