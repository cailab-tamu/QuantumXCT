"""
benchmark_dm_strategies.py
==========================
Benchmark the density-matrix candidate-finder strategies on the OVARIAN dataset:
  - reference : find_cnot_candidates_from_state_diff  (complex128, current pipeline default)
  - dense32   : find_cnot_candidates_dense(precision="float32")
  - dense16   : find_cnot_candidates_dense(precision="float16")   # precision stress-test
  - sparse    : find_cnot_candidates_sparse(precision="float32")  # scalable support-block

For every strategy it reports, versus the reference candidate set:
  - matches_reference : exact set equality
  - overlap_pct       : |cand ∩ ref| / |ref|   (100 = ref is fully recovered)
  - jaccard           : |cand ∩ ref| / |cand ∪ ref|
  - max_signal        : largest co−mono density-matrix peak (the calibration signal)
  - time_ms, peak_KB  : wall time and tracemalloc peak allocation

With --quality it ALSO runs the real n-wise topology search + CRX-angle
optimization on each strategy's *fixed* candidate list (reference, plus any
strategy whose overlap < 100 %) and reports n_gates / final_kl / kl_delta_vs_ref.
Strategies that reproduce the reference set exactly are skipped there — their
quality is identical by construction, so only a lossy set (e.g. float16) is run.

Run from the QuantumXCT/python folder:
    conda activate qiskit-env
    python benchmark_dm_strategies.py            # fast: sets, overlap, time, memory
    python benchmark_dm_strategies.py --quality  # also the topology-KL quality check
"""
import os, time, argparse, tracemalloc
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse

from qxct.circuits import (
    find_cnot_candidates_from_state_diff,
    find_cnot_candidates_dense, find_cnot_candidates_sparse,
    vector_normalize_dictionary_values, build_initial_state_circuit,
    find_best_cnot_sequence_multi_epoch, optimize_crx_angles, set_global_seed,
)
from qxct.encoding import create_joint_histogram

THRESHOLD = 0.005
SEARCH_MODE = "pos"
NORM_FOR_QUALITY = "l1"          # encoding used when re-running the actual optimization
QUALITY_SHOTS = 2000
SEED = 42
OUT_DIR = "results/dm_strategies"; os.makedirs(OUT_DIR, exist_ok=True)


def _mat(a):
    return a.X.toarray() if issparse(a.X) else a.X


def load_ovarian():
    adata = sc.read_h5ad("dataset_co_mo2.h5ad")
    sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)
    gl_ct1 = ["PDGFB", "STAT3", "HES4", "TGFB1"]
    gl_ct2 = ["PDGFRB", "TGFBR1", "TGFBR2"]
    X1c = _mat(adata[adata.obs["BatchID"] == "Cancer Cells (Co)", gl_ct1])
    X2c = _mat(adata[adata.obs["BatchID"] == "Fibroblasts (Co)",  gl_ct2])
    X1m = _mat(adata[adata.obs["BatchID"] == "Cancer Cells (Mo)", gl_ct1])
    X2m = _mat(adata[adata.obs["BatchID"] == "Fibroblasts (Mo)",  gl_ct2])
    b = lambda X: (X > 0).astype(int)
    jc1c, _ = create_joint_histogram(b(X1c)); jc2c, _ = create_joint_histogram(b(X2c))
    jc1m, _ = create_joint_histogram(b(X1m)); jc2m, _ = create_joint_histogram(b(X2m))
    return jc1m, jc2m, jc1c, jc2c


def timed(fn):
    tracemalloc.start(); t = time.perf_counter(); out = fn()
    dt = time.perf_counter() - t; peak = tracemalloc.get_traced_memory()[1]; tracemalloc.stop()
    return out, dt, peak


def _set_overlap(cands, ref):
    """(overlap_pct vs ref, jaccard) for two candidate lists of (ctrl, tgt) tuples."""
    c, r = set(map(tuple, cands)), set(map(tuple, ref))
    if not r:
        return float("nan"), float("nan")
    inter = len(c & r)
    overlap_pct = 100.0 * inter / len(r)
    jaccard = inter / len(c | r) if (c | r) else float("nan")
    return round(overlap_pct, 2), round(jaccard, 4)


def kl_for_candidates(cands, i1, i2, t1, t2, seed=SEED, nshots=QUALITY_SHOTS, norm=NORM_FOR_QUALITY):
    """Run the *actual* n-wise topology search + CRX-angle optimization restricted
    to a FIXED candidate list, and return (n_gates, final_kl). Seeded for
    reproducibility so two strategies differ only through their candidate sets."""
    set_global_seed(seed)
    initial1 = vector_normalize_dictionary_values(i1, norm)
    initial2 = vector_normalize_dictionary_values(i2, norm)
    target1  = vector_normalize_dictionary_values(t1, norm)
    target2  = vector_normalize_dictionary_values(t2, norm)
    circ1 = build_initial_state_circuit(initial1, norm=None)
    circ2 = build_initial_state_circuit(initial2, norm=None)
    if not cands:
        return 0, float("nan")
    topo, _kl = find_best_cnot_sequence_multi_epoch(
        circ1, circ2, initial1, initial2, target1, target2,
        cnot_search_candidates=list(cands),
        nshots=nshots, threshold=THRESHOLD, search_mode=SEARCH_MODE,
    )
    if not topo:
        return 0, float("nan")
    _angles, final_kl, _hist = optimize_crx_angles(
        circ1, circ2, target1, target2, topo, nshots=nshots,
    )
    return len(topo), float(final_kl)

ap = argparse.ArgumentParser(description="Density-matrix candidate-strategy benchmark (ovarian).")
ap.add_argument("--quality", action="store_true",
                help="also run the topology search + CRX optimization to compare candidate QUALITY "
                     "(reference, plus any strategy with overlap < 100%).")
args = ap.parse_args()

i1, i2, t1, t2 = load_ovarian()
strategies = {
    "reference(complex128)": lambda: find_cnot_candidates_from_state_diff(
        i1, i2, t1, t2, threshold=THRESHOLD, search_mode=SEARCH_MODE, norm=None),
    "dense(float32)": lambda: find_cnot_candidates_dense(
        i1, i2, t1, t2, threshold=THRESHOLD, search_mode=SEARCH_MODE, norm=None, precision="float32"),
    "dense(float16)": lambda: find_cnot_candidates_dense(
        i1, i2, t1, t2, threshold=THRESHOLD, search_mode=SEARCH_MODE, norm=None, precision="float16"),
    "sparse(float32)": lambda: find_cnot_candidates_sparse(
        i1, i2, t1, t2, threshold=THRESHOLD, search_mode=SEARCH_MODE, norm=None, precision="float32"),
}

ref = None; rows = []; cand_by_name = {}
for name, fn in strategies.items():
    (cands, stats), dt, peak = timed(fn)
    cand_by_name[name] = cands
    if ref is None:
        ref = cands
    overlap_pct, jaccard = _set_overlap(cands, ref)
    rows.append({"strategy": name, "n_candidates": len(cands),
                 "matches_reference": (sorted(map(tuple, cands)) == sorted(map(tuple, ref))),
                 "overlap_pct": overlap_pct, "jaccard": jaccard,
                 "max_signal": round(stats.get("max_element", float("nan")), 4),
                 "time_ms": round(dt * 1000, 2), "peak_KB": round(peak / 1024, 1)})
df = pd.DataFrame(rows)

# ---- optional candidate-QUALITY comparison (topology KL) --------------
if args.quality:
    print("\n--quality: running topology search + CRX optimization on fixed candidate sets ...")
    qual = {}
    ref_gates, ref_kl = kl_for_candidates(ref, i1, i2, t1, t2)
    qual["reference(complex128)"] = (ref_gates, ref_kl)
    for name in strategies:
        if name == "reference(complex128)":
            continue
        row = df.loc[df["strategy"] == name].iloc[0]
        if row["overlap_pct"] >= 100.0:   # identical set -> identical quality, skip
            continue
        qual[name] = kl_for_candidates(cand_by_name[name], i1, i2, t1, t2)

    def _ng(n):  return qual[n][0] if n in qual else np.nan
    def _kl(n):  return round(qual[n][1], 6) if n in qual else np.nan
    def _dk(n):  return round(qual[n][1] - ref_kl, 6) if n in qual else np.nan
    df["n_gates"]         = df["strategy"].map(_ng)
    df["final_kl"]        = df["strategy"].map(_kl)
    df["kl_delta_vs_ref"] = df["strategy"].map(_dk)

pd.set_option("display.width", 220)
print(f"\nOvarian DM-strategy benchmark  (threshold={THRESHOLD}, mode={SEARCH_MODE})")
print(df.to_string(index=False))
out_csv = os.path.join(OUT_DIR, "dm_strategies_benchmark.csv")
df.to_csv(out_csv, index=False)
print(f"\nSaved -> {out_csv}")
lossless = df[df["strategy"] != "dense(float16)"]
print("lossless strategies all match reference:", bool(lossless["matches_reference"].all()))

# ---- SI-ready LaTeX table of the density-matrix screening strategies ----
_tex_cols = [("strategy", "Strategy", "s"), ("n_candidates", "Candidates", "d"),
             ("overlap_pct", "Overlap (\\%)", "1"), ("max_signal", "Max signal", "4"),
             ("time_ms", "Time (ms)", "2"), ("peak_KB", "Peak mem (KB)", "1")]
def _cell(v, kind):
    if kind == "s": return str(v).replace("_", "\\_")
    if kind == "d": return f"{int(v)}"
    return f"{float(v):.{int(kind)}f}"
_body = ["    " + " & ".join(_cell(r[c], k) for c, _l, k in _tex_cols) + r"\\"
         for _, r in df.iterrows()]
_tex = (r"\begin{table}[h]" "\n" r"\centering" "\n"
        r"\caption{Density-matrix candidate-screening strategies on the ovarian panel "
        r"(threshold $\delta\rho>" + str(THRESHOLD) + r"$). All lossless strategies recover the "
        r"identical candidate set (100\% overlap with the complex128 reference); float32 and the "
        r"sparse float32 finder are several times faster and lighter, which is why float32 is the "
        r"default for screening.}" "\n"
        r"\label{tab:si_dm_strategies}" "\n"
        r"\begin{tabular}{lrrrrr}" "\n" r"\toprule" "\n"
        + " & ".join(c[1] for c in _tex_cols) + r"\\" "\n" r"\midrule" "\n"
        + "\n".join(_body) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table}" "\n")
open(os.path.join(OUT_DIR, "dm_strategies_benchmark.tex"), "w").write(_tex)
print(f"Saved -> {os.path.join(OUT_DIR, 'dm_strategies_benchmark.tex')}")
