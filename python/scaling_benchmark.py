"""
scaling_benchmark.py
====================
Computation-time (and memory) scaling curve for the QuantumXCT pipeline as the
number of features grows.

Sweeps grid sizes written as  n_source + n_target  (left = source cell type,
right = target cell type; total qubits = n_source + n_target) and records, for
each size, the wall-clock time of every pipeline stage plus a memory estimate.
Default path:  1+1, 2+1, 2+2, 3+2, 3+3, ... up to 10+10 (total qubits 2..20).

Data is SYNTHETIC (random binarized expression with a co-vs-basal shift): the
runtime depends on the qubit count (n_source + n_target), not on which specific
genes are used, so synthetic data gives a clean, reproducible scaling curve and
lets us push to the size where the machine runs out of memory.

Memory: up to 12 qubits the candidate screening builds the dense 2^N x 2^N
density-matrix difference (268 MB at N=12).  Above 12 qubits it switches
automatically to the exact threshold-pruned support block
(find_cnot_candidates_sparse; identical candidates), so the old 4^N wall
(~17 GB at 15, 17 TB at 20) no longer applies.  dm_mem_est_GB still reports
what the dense matrix WOULD need, for the SI memory curve.

Time: with --exact, circuits are scored from the exact statevector
(qxct.exact; the per-cell-type state preparations are applied directly
instead of being synthesized), which is what makes 10+10 feasible.  Without
--exact every candidate is scored by shot sampling (Aer), which is the
original setting and becomes the bottleneck beyond ~12 qubits.

Outputs (to ./scaling_out/):
    scaling_results_{sampled|exact}.csv   one row per (n_source, n_target)
    scaling_curve_{sampled|exact}.png     time vs total qubits (separate per scoring mode)

Run:
    conda activate qiskit-env
    python scaling_benchmark.py                 # default sweep
    python scaling_benchmark.py --exact --time-cap-s 7200   # up to 10+10 genes
    python scaling_benchmark.py --scope inter   # LR-only (no DM screening -> no 4^N wall)
"""

import argparse
import csv
import gc
import os
import sys
import time
import traceback

import numpy as np

try:
    import psutil
    _PROC = psutil.Process()
    _TOTAL_RAM = psutil.virtual_memory().total
except Exception:
    psutil = None
    _PROC = None
    _TOTAL_RAM = 16 * 1024**3  # assume 16 GB if psutil missing

from qxct import (QuantumXct, RunConfig, EncodingConfig, BackendConfig,
                  SearchConfig, OptimizerConfig)
from qxct.encoding import create_joint_histogram

OUT_DIR = "results/scaling"


# Synthetic data.  Genes are ON with probability p ~ U(P_RANGE) (sparse, as in
# binarized scRNA-seq), N_CELLS cells per population.  The original setting
# (400 cells, p ~ U(0.25, 0.75)) spreads the cells almost uniformly over 2^n
# states; above ~16 qubits every density-matrix element then falls below the
# screening threshold, no candidates survive and the "runtime" measures an empty
# search.  Pass --n-cells 400 --p-range 0.25 0.75 to reproduce the old curve.
N_CELLS = 2000
P_RANGE = (0.02, 0.25)


def synth_hist(n_genes, n_cells=None, seed=0, shift=0.0):
    """Random binarized expression -> joint histogram over 2**n_genes states."""
    n_cells = N_CELLS if n_cells is None else n_cells
    rng = np.random.default_rng(seed)
    p = np.clip(rng.uniform(*P_RANGE, size=n_genes) + shift, 0.01, 0.97)
    X = (rng.random((n_cells, n_genes)) < p).astype(int)
    hist, _ = create_joint_histogram(X)
    return hist


def dm_bytes(total_qubits):
    """Bytes for the 2^N x 2^N complex128 density matrix built during screening."""
    return (2 ** (2 * total_qubits)) * 16


def default_pairs(max_qubits):
    """1+1, 2+1, 2+2, 3+2, ... incrementing total qubits by 1 each step."""
    pairs = []
    for t in range(2, max_qubits + 1):
        n1 = (t + 1) // 2   # source (ceil)
        n2 = t // 2         # target (floor)
        pairs.append((n1, n2))
    return pairs


def rss_gb():
    if _PROC is not None:
        return _PROC.memory_info().rss / 1e9
    return float("nan")


def run_point(n1, n2, scope, shots, n_epochs, max_greedy_depth, threshold, kl_tol, seed=42, exact=False):
    """Run the pipeline once at size (n1, n2); return a dict of stage timings.
    exact=True scores every circuit from the exact statevector (no shots)."""
    from qxct.exact import use_exact_scoring
    from qxct import circuits as _circ, analysis as _ana
    _prev = (_circ.score_circuit_kl_divergences, _ana.score_circuit_kl_divergences)
    if exact:
        use_exact_scoring(True)
    try:
        return _run_point(n1, n2, scope, shots, n_epochs, max_greedy_depth, threshold, kl_tol, seed, exact)
    finally:
        _circ.score_circuit_kl_divergences, _ana.score_circuit_kl_divergences = _prev


def _run_point(n1, n2, scope, shots, n_epochs, max_greedy_depth, threshold, kl_tol, seed, exact):
    N = n1 + n2
    jc1_mo = synth_hist(n1, seed=seed,     shift=0.0)
    jc2_mo = synth_hist(n2, seed=seed + 1, shift=0.0)
    jc1_co = synth_hist(n1, seed=seed,     shift=0.15)   # same seed + shift -> a real co-vs-mono difference
    jc2_co = synth_hist(n2, seed=seed + 1, shift=0.15)

    cfg = RunConfig(
        encoding=EncodingConfig(norm="l1"),
        backend=BackendConfig(shots=shots),
        search=SearchConfig(algorithm="multi_epoch", candidate_scope=scope,
                            threshold=threshold, search_mode="abs",
                            n_epochs=n_epochs, kl_tol=kl_tol,
                            max_greedy_depth=max_greedy_depth),
        optimizer=OptimizerConfig(method="COBYLA", initial_angle=np.pi / 2),
        seed=seed,
    )
    qx = QuantumXct(cfg)

    rss0 = rss_gb()
    t = time.perf_counter(); qx.encode(jc1_mo, jc2_mo, jc1_co, jc2_co); t_enc = time.perf_counter() - t
    t = time.perf_counter(); cands = qx.screen_candidates();            t_scr = time.perf_counter() - t
    t = time.perf_counter()
    if scope == "intra":
        qx.search_topology_grn_sequential()
    else:
        qx.search_topology()
    t_srch = time.perf_counter() - t
    t = time.perf_counter(); qx.optimize_angles();                      t_opt = time.perf_counter() - t
    rss1 = rss_gb()

    return {
        "n_source": n1, "n_target": n2, "total_qubits": N,
        "n_candidates": len(cands), "n_gates": len(qx.result.topology),
        "t_encode_s": round(t_enc, 4), "t_screen_s": round(t_scr, 4),
        "t_search_s": round(t_srch, 4), "t_optimize_s": round(t_opt, 4),
        "t_total_s": round(t_enc + t_scr + t_srch + t_opt, 4),
        "dm_mem_est_GB": round(dm_bytes(N) / 1e9, 4),
        "rss_delta_GB": round(rss1 - rss0, 4) if rss0 == rss0 else float("nan"),
        "scoring": "exact" if exact else f"{shots} shots",
        "status": "ok",
    }

ap = argparse.ArgumentParser()
ap.add_argument("--max-qubits", type=int, default=20, help="largest n_source+n_target to attempt")
ap.add_argument("--scope", default="all", choices=["all", "intra", "inter"],
                help="'all'/'intra' use 4^N density screening (memory wall); 'inter' skips it")
ap.add_argument("--shots", type=int, default=1000)
ap.add_argument("--n-epochs", type=int, default=2, help="held small so the curve reflects size, not epoch count")
ap.add_argument("--max-greedy-depth", type=int, default=4)
ap.add_argument("--threshold", type=float, default=0.002)
ap.add_argument("--kl-tol", type=float, default=0.01)
ap.add_argument("--mem-fraction", type=float, default=0.6,
                help="stop before predicted screening memory exceeds this fraction of RAM")
ap.add_argument("--allow-oom", action="store_true",
                help="ignore the memory prediction and push into the wall (may crash)")
ap.add_argument("--time-cap-s", type=float, default=600.0,
                help="stop the sweep once a single point exceeds this wall-time")
ap.add_argument("--seed", type=int, default=42, help="master seed for reproducibility")
ap.add_argument("--exact", action="store_true", help="exact statevector scoring (needed for 10+10)")
ap.add_argument("--n-cells", type=int, default=N_CELLS, help="cells per synthetic population")
ap.add_argument("--p-range", type=float, nargs=2, default=list(P_RANGE), help="gene ON-probability range")
ap.add_argument("--plot-only", action="store_true",
                help="skip the sweep; just (re)draw the curve from the existing scaling_results CSV")
args = ap.parse_args()
N_CELLS, P_RANGE = args.n_cells, tuple(args.p_range)

np.random.seed(args.seed)
os.makedirs(OUT_DIR, exist_ok=True)
_sfx = "exact" if args.exact else "sampled"
csv_path = os.path.join(OUT_DIR, f"scaling_results_{_sfx}.csv")
fields = ["n_source", "n_target", "total_qubits", "n_candidates", "n_gates",
          "t_encode_s", "t_screen_s", "t_search_s", "t_optimize_s", "t_total_s",
          "dm_mem_est_GB", "rss_delta_GB", "scoring", "status"]


def _plot(rows, sfx):
    """Scaling plot of the per-stage WALL-CLOCK time vs total qubits: density-matrix
    SCREENING (pruning), topology SEARCH and angle OPTIMIZATION plotted separately,
    plus the total. Memory is NOT plotted -- it goes to a table (see _write_table)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _f(r, k):
        try:
            return float(r[k])
        except Exception:
            return float("nan")
    ok = [r for r in rows if str(r.get("status")) == "ok"]
    if not ok:
        print("  (no successful points to plot)")
        return
    N = [_f(r, "total_qubits") for r in ok]
    series = [
        ("t_screen_s",   "density-matrix screening (pruning)", "#eb6834", "-o"),
        ("t_search_s",   "topology search",                    "#2a78d6", "-o"),
        ("t_optimize_s", "angle optimization",                 "#1baf7a", "-s"),
        ("t_total_s",    "total",                              "#4a3aa7", "--"),
    ]
    fig, ax = plt.subplots(figsize=(6.0, 3.9))
    for key, lab, col, style in series:
        ax.plot(N, [_f(r, key) for r in ok], style, color=col, ms=4, lw=1.6, label=lab)
    ax.set_xlabel("total qubits (source + target genes)")
    ax.set_ylabel("wall-clock time (s)")
    ax.set_yscale("log"); ax.grid(True, ls=":", alpha=0.5)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(f"QuantumXCT scaling: screening vs search ({sfx} scoring)")
    fig.tight_layout()
    base = os.path.join(OUT_DIR, f"scaling_curve_{sfx}")
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  plot -> {base}.png / .pdf")
    _write_table(ok, sfx)


def _write_table(ok, sfx):
    """Per-stage time + density-matrix memory table (the memory that used to be on the
    plot now lives here). Writes scaling_table_{sfx}.csv and .tex."""
    import csv as _csv

    def _f(r, k):
        try:
            return float(r[k])
        except Exception:
            return float("nan")
    cols = [("total_qubits", "Qubits", 0), ("n_candidates", "Candidates", 0),
            ("t_screen_s", "Screening (s)", 3), ("t_search_s", "Search (s)", 3),
            ("t_optimize_s", "Angle opt. (s)", 3), ("t_total_s", "Total (s)", 3),
            ("dm_mem_est_GB", "DM memory (GB)", 3)]
    csv_out = os.path.join(OUT_DIR, f"scaling_table_{sfx}.csv")
    with open(csv_out, "w", newline="") as f:
        w = _csv.writer(f); w.writerow([c[1] for c in cols])
        for r in ok:
            w.writerow([f"{_f(r, k):.{nd}f}" if nd else int(_f(r, k)) for k, _lab, nd in cols])
    body = []
    for r in ok:
        body.append("    " + " & ".join(
            (f"{int(_f(r, k))}" if nd == 0 else f"{_f(r, k):.{nd}f}") for k, _lab, nd in cols) + r"\\")
    tex = (r"\begin{table}[h]" "\n" r"\centering" "\n"
           r"\caption{Per-stage wall-clock time and density-matrix memory of QuantumXCT versus "
           r"problem size (" + sfx + r" scoring). Screening is the density-matrix candidate pruning; "
           r"above 12 qubits it uses the sparse float32 finder, which keeps it cheap while the dense "
           r"$4^N$ memory estimate grows.}" "\n"
           r"\label{tab:si_scaling_" + sfx + r"}" "\n"
           r"\begin{tabular}{rrrrrrr}" "\n" r"\toprule" "\n"
           + " & ".join(c[1] for c in cols) + r"\\" "\n" r"\midrule" "\n"
           + "\n".join(body) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table}" "\n")
    open(os.path.join(OUT_DIR, f"scaling_table_{sfx}.tex"), "w").write(tex)
    print(f"  table -> {csv_out} / .tex")


if args.plot_only:
    import csv as _csv
    with open(csv_path) as _fh:
        _rows = list(_csv.DictReader(_fh))
    _plot(_rows, _sfx)
    sys.exit(0)

_scoring = "EXACT statevector (shot-noise-free)" if args.exact else f"Aer shot sampling ({args.shots} shots)"
print("=" * 78)
print(f"SCORING MODE: {_scoring}")
if args.exact:
    print("  NOTE: this curve measures the runtime of EXACT statevector scoring, which is")
    print("        NOT the shot-sampling pipeline. It is the variant that makes 10+10 feasible;")
    print("        report it as such (see the 'scoring' column in the CSV).")
print("=" * 78)
print(f"RAM detected: {_TOTAL_RAM/1e9:.1f} GB | scope={args.scope} | "
      f"stopping if screening would exceed {args.mem_fraction*100:.0f}% of RAM"
      + ("" if args.scope == 'inter' else " (density screening ~ 4^N)"))
print(f"{'size':>8} {'qubits':>6} {'cand':>5} {'gates':>5} "
      f"{'screen(s)':>10} {'search(s)':>10} {'total(s)':>9} {'DMmem(GB)':>10}")

rows = []
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    for (n1, n2) in default_pairs(args.max_qubits):
        N = n1 + n2
        # memory guard (only meaningful when screening is used)
        if args.scope != "inter" and N <= 12:          # dense screening only up to 12 qubits
            est = dm_bytes(N)
            if not args.allow_oom and est > args.mem_fraction * _TOTAL_RAM:
                row = {k: "" for k in fields}
                row.update({"n_source": n1, "n_target": n2, "total_qubits": N,
                            "dm_mem_est_GB": round(est / 1e9, 2),
                            "status": "stopped: predicted OOM"})
                writer.writerow(row); f.flush(); rows.append(row)
                print(f"  {n1}+{n2:<5} {N:>6}  --  predicted screening memory "
                      f"{est/1e9:.1f} GB > {args.mem_fraction*100:.0f}% RAM -> stopping "
                      f"(use --allow-oom or --scope inter to push further)")
                break
        try:
            r = run_point(n1, n2, args.scope, args.shots, args.n_epochs,
                          args.max_greedy_depth, args.threshold, args.kl_tol, seed=args.seed,
                          exact=args.exact)
            writer.writerow(r); f.flush(); rows.append(r)
            print(f"  {n1}+{n2:<5} {N:>6} {r['n_candidates']:>5} {r['n_gates']:>5} "
                  f"{r['t_screen_s']:>10} {r['t_search_s']:>10} {r['t_total_s']:>9} "
                  f"{r['dm_mem_est_GB']:>10}")
            gc.collect()
            if r["t_total_s"] > args.time_cap_s:
                print(f"  time cap ({args.time_cap_s}s) exceeded — stopping sweep.")
                break
        except MemoryError:
            row = {k: "" for k in fields}
            row.update({"n_source": n1, "n_target": n2, "total_qubits": N, "status": "MemoryError"})
            writer.writerow(row); f.flush(); rows.append(row)
            print(f"  {n1}+{n2}  -> MemoryError (the wall). Stopping.")
            break
        except Exception as e:
            row = {k: "" for k in fields}
            row.update({"n_source": n1, "n_target": n2, "total_qubits": N,
                        "status": f"error: {type(e).__name__}"})
            writer.writerow(row); f.flush(); rows.append(row)
            print(f"  {n1}+{n2}  -> {type(e).__name__}: {e}")
            traceback.print_exc()
            break

print(f"\nCSV written to {csv_path}")
_plot(rows, _sfx)
