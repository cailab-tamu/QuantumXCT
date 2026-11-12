"""Regenerate python/CODE_MAP.txt.

For every module/script it lists each top-level function/class (and methods) with its
line range and a provenance tag, comparing three snapshots of the code:

  HEAD        = last git commit (git show HEAD:python/<file>)
  PRE-REVIEW  = the working copy before the review session
                (Claude outputs/_snapshot_python_for_review.tgz)
  CURRENT     = the files on disk now

Tags
  [unchanged]          identical in HEAD, PRE-REVIEW and CURRENT
  [pre-review NEW]     not in HEAD, added before the review session (earlier Cowork work)
  [pre-review MOD]     in HEAD, changed before the review session
  [review NEW]         added during the review session
  [review MOD]         changed during the review session

Run from QuantumXCT/python:  python analysis_tools/make_code_map.py
"""
import ast, datetime, io, os, subprocess, tarfile

HERE = os.getcwd()
SNAP = os.path.join(HERE, "..", "..", "Claude outputs", "_snapshot_python_for_review.tgz")

FILES = (sorted("qxct/" + f for f in os.listdir("qxct") if f.endswith(".py"))
         + ["run_all.py", "run_hardware_postopt.py", "collector_ccc_comparison.py",
            "scaling_benchmark.py", "benchmark_dm_strategies.py", "clean_outputs.py",
            "export_notebooks_to_py.py", "simCellsSpace/simulate_cells.py"]
         + sorted("analysis_tools/" + f for f in os.listdir("analysis_tools") if f.endswith(".py") and f != "__init__.py")
         + sorted("tests/" + f for f in os.listdir("tests") if f.endswith(".py")))


def head_src(path):
    try:
        return subprocess.run(["git", "show", f"HEAD:python/{path}"], cwd=HERE, capture_output=True,
                              text=True, check=True).stdout
    except Exception:
        return None


def snap_src(path, _cache={}):
    if "tar" not in _cache:
        _cache["tar"] = tarfile.open(SNAP) if os.path.exists(SNAP) else None
    t = _cache["tar"]
    if t is None:
        return None
    try:
        return t.extractfile("python/" + path).read().decode("utf-8", "replace")
    except Exception:
        return None


def defs(src):
    out = {}
    if src is None:
        return out
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return out
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            kind = "class" if isinstance(n, ast.ClassDef) else "def"
            out[n.name] = (kind, n.lineno, n.end_lineno, ast.get_source_segment(src, n))
            if kind == "class":
                for m in n.body:
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        out[f"{n.name}.{m.name}"] = ("meth", m.lineno, m.end_lineno, ast.get_source_segment(src, m))
    return out


def same_code(a, b):
    """True if two source segments differ only in comments/whitespace (same AST)."""
    import textwrap
    try:
        return ast.dump(ast.parse(textwrap.dedent(a))) == ast.dump(ast.parse(textwrap.dedent(b)))
    except SyntaxError:
        return False


def tag(name, cur, pre, head):
    c = cur[name][3]
    p = pre.get(name, (None,) * 4)[3]
    h = head.get(name, (None,) * 4)[3]
    if p is None:                       # did not exist before the review
        return "[review NEW]"
    if p != c:
        return "[review MOD, comments only]" if same_code(p, c) else "[review MOD]"
    if h is None:
        return "[pre-review NEW]"
    if h != p:
        return "[pre-review MOD, comments only]" if same_code(h, p) else "[pre-review MOD]"
    return "[unchanged]"


def file_tag(path, cur_src, pre_src, head_src_):
    if pre_src is None:
        return "REVIEW-NEW FILE"
    if head_src_ is None:
        return "PRE-REVIEW-NEW FILE" + ("" if pre_src == cur_src else ", modified in review")
    parts = []
    parts.append("unchanged since HEAD" if head_src_ == pre_src else "modified before review")
    if pre_src != cur_src:
        parts.append("modified in review")
    return ", ".join(parts)


lines = [
    "QuantumXCT - CODE MAP",
    f"generated {datetime.date.today()} by analysis_tools/make_code_map.py",
    "Snapshots compared: HEAD (git) -> PRE-REVIEW (working copy before the review) -> CURRENT.",
    "Tags: [unchanged] | [pre-review NEW] / [pre-review MOD] = earlier Cowork work, not yet committed |",
    "      [review NEW] / [review MOD] = added/changed during the review session.",
    "      '..., comments only' = same code (identical AST), only comments/whitespace differ.",
    "Line numbers refer to the CURRENT files.",
    "=" * 96,
]
counts = {}
for f in FILES:
    if not os.path.exists(f):
        continue
    cur_src = open(f, encoding="utf-8").read()
    pre_src, h_src = snap_src(f), head_src(f)
    cur, pre, head = defs(cur_src), defs(pre_src), defs(h_src)
    lines.append(f"\n### {f}   ({cur_src.count(chr(10))} lines)   [{file_tag(f, cur_src, pre_src, h_src)}]")
    for name, (kind, a, b, _) in cur.items():
        t = tag(name, cur, pre, head)
        counts[t] = counts.get(t, 0) + 1
        label = ("    ." + name.split(".", 1)[1]) if kind == "meth" else f"{kind} {name}"
        lines.append(f"   {label:<50} L{a:>5}-{b:<5} {t}")
    gone = [n for n in pre if n not in cur]
    for n in gone:
        lines.append(f"   {n:<50} {'':13} [removed in review]")

lines += ["", "=" * 96, "SUMMARY OF TAGS (functions/classes/methods)"]
for k in sorted(counts):
    lines.append(f"   {k:<20} {counts[k]}")

lines += ["", "=" * 96, "run_all.py STEPS  (python run_all.py --list)", ""]
import importlib.util
spec = importlib.util.spec_from_file_location("run_all", "run_all.py")
ra = importlib.util.module_from_spec(spec); spec.loader.exec_module(ra)
orig = {"sim", "ovarian", "scaling", "dm", "tests"}
for key, kind, path, desc in ra.STEPS:
    args = " ".join(ra.STEP_ARGS.get(key, [])) if hasattr(ra, "STEP_ARGS") else ""
    origin = "original step" if key in orig else "added in review"
    lines.append(f"   {key:<11} {kind:<8} {path} {args}".rstrip())
    lines.append(f"   {'':<11} {desc}  [{origin}]")

lines += ["", "=" * 96, "REVISION OUTPUTS  (results/analyses unless stated)", "",
          "   exact      -> exact_summary.json, *_exact_contributions.csv, *_random_null_{all,inter,intra}.csv",
          "   table2     -> table2_exact_audit_L2.csv                     (main Table 3; SI Fig S3)",
          "   scopes_*   -> {ovarian,synthetic}_exact_scopes_norms.csv   (main Tables 2 and 4; SI Table S1)",
          "   boot_*     -> ovarian_boot.csv, sim_boot.csv               (bootstrap resamples; dropped from the revised SI)",
          "   budget     -> budget_capacity_table.{csv,tex,xlsx}          (SI method-comparison table)",
          "   benchAB    -> benchA_benchB_exact/*                        (SI Table S4; Benchmark B paragraph)",
          "   algos      -> ovarian_algorithms_thr002_exact.csv          (algorithm comparison incl. VQE)",
          "   post       -> synthetic_recovery_corrected.csv, synthetic_single_edge_scores.csv,",
          "                 ovarian_bootstrap_{hub,target}_frequency.csv (bootstrap stability; dropped from the revised SI)",
          "   hwdry      -> results/hardware/hardware_postopt_report.json (SI Section S10, Fig S8)",
          "   ccc        -> results/ccc_comparison/ccc_comparison_table.csv",
          "   figs       -> figures/figS*.pdf",
          "   searchfigs -> results/analyses/search_panels/{ds}_{norm}_{sampled|exact}/*, search_figures/*_{sampled|exact}_summary.csv",
          "   scopes_*   also write {ovarian,synthetic}_exact_scopes_algorithms.csv (multi_epoch + iterative_n_wise)",
          "   paper      -> copies figures to QuantumXCT_KL_paper/revision and rebuilds the PDFs",
          "   verify     -> exit code 0 only if every quoted number matches the results",
          "   codemap    -> CODE_MAP.txt (this file)"]
open("CODE_MAP.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("CODE_MAP.txt written;", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
