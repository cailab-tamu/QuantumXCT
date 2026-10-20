"""
run_all.py
==========
Run EVERYTHING that produces the QuantumXCT results, in dependency order, from
one command. Executes the canonical result set:

  1. qxct_pred_classic_sim_oo.ipynb   simulated cells: intra/inter/all scopes,
                                       edge/direction recovery, L1-vs-L2 encoding
  2. qxct_ovarian_cocultured_oo.ipynb  merged ovarian: single-pipeline demo +
                                       benchmark A + angle curves + benchmark B +
                                       L1/L2 + algorithm comparison
  3. scaling_benchmark.py             N-scaling / dense-vs-sparse memory & time,
                                       run twice: SAMPLED (step 'scaling') and EXACT
                                       ('scaling_exact') -> scaling_results_{sampled,exact}.csv
  4. benchmark_dm_strategies.py       ovarian DM candidate-finder strategies
                                       (add --quality for topology-KL check)
  5. tests/test_dm_strategies.py      candidate-equivalence validation (pytest)

Notebooks are executed IN PLACE (outputs saved back into the .ipynb), and each
notebook/script also writes its own figures/tables under results/<...>/.

Usage (from the QuantumXCT/python folder, in your qiskit-env):
    conda activate qiskit-env
    python run_all.py                 # run the whole canonical set, keep going on error
    python run_all.py --stop-on-error # abort at the first failure
    python run_all.py --only sim ovarian     # run just these steps (by key)
    python run_all.py --skip scaling         # run everything except a step
    python run_all.py --list                 # show the steps and exit
    python run_all.py --timeout 10800        # impose a 3h per-CELL cap (default: no cap)
    python run_all.py --quality              # pass --quality to the DM benchmark step

Step keys: sim, ovarian, scaling, scaling_exact, dm, exact, table2, scopes_ov, scopes_sim,
boot_ov, boot_sim, budget, benchAB, algos, post, hwdry, ccc, figs, anglefig,
anglefig_exact, datahist, searchfigs, searchfigs_exact, paper, verify, tests, codemap

Figure scoring: the paper/poster figures are SAMPLED (real Aer, 5000 shots) — that is
the `anglefig` and `searchfigs` steps. Their EXACT statevector counterparts (shot-noise-
free, for SI verification only) are `anglefig_exact` and `searchfigs_exact`, written to
separate *_exact files/folders so they never overwrite the sampled ones. Running the whole
set produces both; the contribution/scope/bootstrap tables (exact, table2,
scopes_*, boot_*, budget, benchAB, algos, post) stay exact by design (SI Section S4).

Analyses only (fast path, ~30-90 min without the bootstraps):
    python run_all.py --only exact table2 scopes_ov scopes_sim budget benchAB algos post hwdry figs paper verify tests
Everything (the two notebooks and the bootstraps are the slow steps):
    python run_all.py
"""
import argparse
import importlib.util
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
LOG_PATH = os.path.join(RESULTS, "run_all_log.txt")

# key -> (kind, path, human description). kind in {notebook, script, pytest}
STEPS = [
    ("sim",     "notebook", "qxct_pred_classic_sim_oo.ipynb",
     "Simulated cells: scopes (intra/inter/all), edge+direction recovery, L1-vs-L2"),
    ("ovarian", "notebook", "qxct_ovarian_cocultured_oo.ipynb",
     "Merged ovarian: pipeline demo, benchmark A, angle curves, benchmark B, L1/L2, algo comparison"),
    ("scaling", "script", "scaling_benchmark.py",
     "N-scaling runtime/memory, SAMPLED Aer (2 epochs, depth 4, 1000 shots; SI S9) -> scaling_results_sampled.csv / scaling_curve_sampled.png"),
    ("scaling_exact", "script", "scaling_benchmark.py",
     "N-scaling, EXACT statevector (reaches 10+10 genes) -> scaling_results_exact.csv / scaling_curve_exact.png"),
    ("dm",      "script",   "benchmark_dm_strategies.py",
     "Ovarian DM candidate-finder strategy benchmark (overlap/jaccard/time/memory)"),
    # ---- exact-scoring analyses: contributions, nulls, bootstrap, baselines ----
    ("exact",   "script",   os.path.join("analysis_tools", "check_exact.py"),
     "Exact contributions, shot noise, local-RY and random-topology nulls (needs sim + ovarian runs)"),
    ("table2",  "script",   os.path.join("analysis_tools", "table2_audit.py"),
     "Exact audit of the manuscript Table 2 (L2) -> main Table 3"),
    ("scopes_ov", "script", os.path.join("analysis_tools", "run_search_exact.py"),
     "Exact search: ovarian scopes x L1/L2"),
    ("scopes_sim", "script", os.path.join("analysis_tools", "run_search_exact.py"),
     "Exact search: synthetic scopes x L1/L2"),
    ("boot_ov", "script",   os.path.join("analysis_tools", "run_search_exact.py"),
     "Bootstrap of cells, ovarian (NBOOT env, default 50; slow)"),
    ("boot_sim", "script",  os.path.join("analysis_tools", "run_search_exact.py"),
     "Bootstrap of cells, synthetic (NBOOT env, default 50; slow)"),
    ("budget",  "script",   os.path.join("analysis_tools", "run_budget.py"),
     "Budget-matched Benchmark B v2 (effective degrees of freedom)"),
    ("benchAB", "script",   os.path.join("analysis_tools", "run_scopes_and_benchB.py"),
     "Benchmark A (per-scope ablation) + Benchmark B (best-effort, D=1/D=2), exact scoring"),
    ("algos",   "script",   os.path.join("analysis_tools", "run_vqe.py"),
     "Algorithm comparison incl. VQE-hybrid at threshold 0.02"),
    ("post",    "script",   os.path.join("analysis_tools", "post_tables.py"),
     "Edge recovery (corrected ground truth), single-edge scores, bootstrap hub/target tables"),
    ("hwdry",   "script",   "run_hardware_postopt.py",
     "Post-optimization circuit: exact + noisy-device dry run (add --hardware to submit)"),
    ("ccc",     "script",   "collector_ccc_comparison.py",
     "Collect CellChat / LIANA / scTenifoldXct results (run the tool scripts in their envs first)"),
    ("figs",    "script",   os.path.join("analysis_tools", "make_si_figures.py"),
     "Supplementary figures -> results/analyses/figures"),
    ("anglefig", "script",  os.path.join("analysis_tools", "run_angle_init.py"),
     "Angle-initialization study (0 vs pi/2), synthetic + ovarian, SAMPLED -> *_angle_init_curves_sampled + angle_init_study_sampled.csv"),
    ("anglefig_exact", "script",  os.path.join("analysis_tools", "run_angle_init.py"),
     "Angle-initialization study, EXACT statevector (SI verification) -> *_angle_init_curves_exact + angle_init_study_exact.csv"),
    ("datahist", "script",  os.path.join("analysis_tools", "make_data_histograms.py"),
     "Data joint histograms (Frequency, bit strings; raw counts), synthetic + ovarian -> *_data_hist_*"),
    ("searchfigs", "script", os.path.join("analysis_tools", "make_search_figures.py"),
     "N-wise / multi-epoch / VQE+multi-epoch x GRN / LR / GRN+LR figures, SAMPLED pipeline (circuits, "
     "networks, angle traces, state occupancy) -> results/analyses/search_panels/{ds}_{norm}_sampled"),
    ("searchfigs_exact", "script", os.path.join("analysis_tools", "make_search_figures.py"),
     "Same figures, EXACT statevector scoring (SI verification) -> results/analyses/search_panels/{ds}_{norm}_exact"),
    ("paper",   "script",   os.path.join("analysis_tools", "sync_to_paper.py"),
     "Copy SI figures into QuantumXCT_KL_paper/revision and rebuild the PDFs"),
    ("verify",  "script",   os.path.join("analysis_tools", "verify_numbers.py"),
     "Check every number quoted in the revised manuscript/SI against the results"),
    ("tests",   "pytest",   "tests",
     "Integrity tests (encodings, bit order, exact scorer, candidate finders, sampler options)"),
    ("codemap", "script",   os.path.join("analysis_tools", "make_code_map.py"),
     "Regenerate CODE_MAP.txt (functions, line ranges, provenance tags)"),
]

# extra command-line arguments for script steps
STEP_ARGS = {"scopes_ov": ["ovarian_basic"], "scopes_sim": ["sim_basic"],
             "boot_ov": ["ovarian_boot"], "boot_sim": ["sim_boot"],
             "anglefig_exact": ["--exact"], "searchfigs_exact": ["--exact"],
             "scaling_exact": ["--exact", "--time-cap-s", "7200"]}

# Datasets each step needs; warn (don't abort) if missing.
DATASET_REQ = {
    "sim":     [os.path.join("simCellsSpace", "sim_cells_datasets_co_mo_simulated_classical.h5ad")],
    "ovarian": ["dataset_co_mo2.h5ad"],
    "dm":      ["dataset_co_mo2.h5ad"],
}


def _have(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def preflight():
    """Report environment readiness. Returns list of warning strings (non-fatal)."""
    warns = []
    if not _have("qxct"):
        warns.append("Package 'qxct' is not importable from here — run from the QuantumXCT/python folder "
                     "with your qiskit-env active.")
    if not _have("nbconvert"):
        warns.append("'nbconvert' not found — notebook steps will fail. Install: pip install nbconvert ipykernel")
    if not _have("qiskit"):
        warns.append("'qiskit' not found — the quantum steps need it (conda activate qiskit-env).")
    for key, files in DATASET_REQ.items():
        for f in files:
            if not os.path.exists(os.path.join(HERE, f)):
                warns.append(f"Dataset for step '{key}' missing: {f}")
    return warns


def run_notebook(path: str, timeout: int) -> subprocess.CompletedProcess:
    # timeout <= 0  -> no per-cell limit (-1), i.e. let every cell run to completion.
    # The heavy quantum cells (esp. the vqe_hybrid algorithm comparison) can run for
    # hours; a per-cell cap is opt-in via --timeout N.
    cell_timeout = timeout if timeout and timeout > 0 else -1
    cmd = [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
           "--execute", "--inplace",
           f"--ExecutePreprocessor.timeout={cell_timeout}", path]
    return subprocess.run(cmd, cwd=HERE)


def run_script(path: str, extra_args) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = HERE + os.pathsep + env.get("PYTHONPATH", "")   # analysis_tools import qxct / analysis_tools
    env.setdefault("MPLBACKEND", "Agg")
    return subprocess.run([sys.executable, path, *extra_args], cwd=HERE, env=env)


def run_pytest(path: str) -> subprocess.CompletedProcess:
    if _have("pytest"):
        return subprocess.run([sys.executable, "-m", "pytest", path, "-q"], cwd=HERE)
    # the test files use pytest fixtures/parametrize, so they cannot run without it
    print("[run_all] pytest is not installed in this environment -> tests NOT run.\n"
          "          Install it with:  pip install pytest   (then: python run_all.py --only tests)")
    return subprocess.CompletedProcess(args=[path], returncode=1)


def main():
    ap = argparse.ArgumentParser(description="Run all QuantumXCT result-producing notebooks and scripts.")
    ap.add_argument("--only", nargs="+", metavar="KEY", help="run only these step keys")
    ap.add_argument("--skip", nargs="+", metavar="KEY", default=[], help="skip these step keys")
    ap.add_argument("--stop-on-error", action="store_true", help="abort at the first failing step")
    ap.add_argument("--timeout", type=int, default=0,
                    help="per-CELL execution timeout in seconds; 0 (default) = no limit, "
                         "let every cell finish (the vqe_hybrid comparison can take hours)")
    ap.add_argument("--quality", action="store_true", help="pass --quality to the DM-benchmark step")
    ap.add_argument("--list", action="store_true", help="list the steps and exit")
    ap.add_argument("--hardware", action="store_true", help="hwdry step: also submit to a real IBM device")
    args = ap.parse_args()

    valid = {k for k, *_ in STEPS}
    if args.list:
        print("Available steps (key : description):")
        for k, kind, path, desc in STEPS:
            print(f"  {k:8s} [{kind:8s}] {path}\n           {desc}")
        return 0

    for bad in (args.only or []) + args.skip:
        if bad not in valid:
            print(f"Unknown step key: {bad}. Valid keys: {', '.join(sorted(valid))}")
            return 2

    selected = [s for s in STEPS
                if (args.only is None or s[0] in args.only) and s[0] not in args.skip]
    if not selected:
        print("No steps selected.")
        return 0

    os.makedirs(RESULTS, exist_ok=True)
    print("=" * 78)
    print("QuantumXCT — run_all")
    print(f"  python : {sys.executable}")
    print(f"  folder : {HERE}")
    print(f"  steps  : {', '.join(s[0] for s in selected)}")
    print("=" * 78)

    warns = preflight()
    if warns:
        print("\nPreflight warnings:")
        for w in warns:
            print(f"  ! {w}")
        print("  (continuing anyway — fix these if a step fails)\n")

    log_lines = [f"run_all @ {time.strftime('%Y-%m-%d %H:%M:%S')}", f"python: {sys.executable}", ""]
    results = []
    overall_t0 = time.time()

    for key, kind, path, desc in selected:
        full = os.path.join(HERE, path)
        print("\n" + "-" * 78)
        print(f">>> [{key}] {kind}: {path}")
        print(f"    {desc}")
        print("-" * 78, flush=True)
        if not os.path.exists(full):
            print(f"    SKIPPED — file not found: {path}")
            results.append((key, path, "MISSING", 0.0))
            log_lines.append(f"[{key}] MISSING ({path})")
            if args.stop_on_error:
                break
            continue

        t0 = time.time()
        try:
            if kind == "notebook":
                cp = run_notebook(path, args.timeout)
            elif kind == "pytest":
                cp = run_pytest(path)
            else:  # script
                extra = ["--quality"] if (key == "dm" and args.quality) else list(STEP_ARGS.get(key, []))
                if key == "hwdry" and args.hardware:
                    extra.append("--hardware")
                cp = run_script(path, extra)
            rc = cp.returncode
        except Exception as e:  # noqa: BLE001
            print(f"    ERROR launching step: {e}")
            rc = -1
        dt = time.time() - t0
        status = "OK" if rc == 0 else f"FAIL(rc={rc})"
        print(f"    -> {status}  ({dt:.1f}s)")
        results.append((key, path, status, dt))
        log_lines.append(f"[{key}] {status}  {dt:.1f}s  ({path})")
        if rc != 0 and args.stop_on_error:
            print("\nStopping (--stop-on-error).")
            break

    total = time.time() - overall_t0
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    width = max((len(k) for k, *_ in results), default=4)
    for key, path, status, dt in results:
        print(f"  {key:<{width}}  {status:<12}  {dt:8.1f}s   {path}")
    print("-" * 78)
    print(f"  total: {total:.1f}s")
    n_fail = sum(1 for _, _, s, _ in results if not s.startswith("OK"))
    print(f"  {len(results)} steps, {n_fail} not-OK")

    log_lines.append("")
    log_lines.append(f"total {total:.1f}s, {n_fail} not-OK")
    try:
        with open(LOG_PATH, "w", encoding="utf-8") as fh:
            fh.write("\n".join(log_lines) + "\n")
        print(f"\nLog -> {LOG_PATH}")
    except OSError as e:
        print(f"\n(could not write log: {e})")

    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
