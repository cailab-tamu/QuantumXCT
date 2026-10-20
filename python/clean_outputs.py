#!/usr/bin/env python3
"""clean_outputs.py — tidy the QuantumXCT/python folder.

DRY-RUN by default (prints what it WOULD do); pass --apply to act. Nothing is
ever deleted: legacy outputs, stray figures and superseded files are MOVED under
results/ so you can review, then delete by hand. A final "REVIEW MANUALLY"
section only LISTS likely-redundant items (duplicate/old drafts) — it never
moves or deletes them, so you decide.

    python clean_outputs.py                 # preview
    python clean_outputs.py --apply         # perform the moves
    python clean_outputs.py --apply --pycache   # also delete __pycache__ dirs
"""
import argparse
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

# legacy output folder  ->  new home under results/
FOLDER_MAP = {
    "benchmark_out_ovarian_cancer": "benchmark_ovarian",
    "benchmark_out":                "benchmark_grn_lr",
    "benchmark_out_classic_sim":    "classic_sim",
    "benchmark_out_sc":             "sc_cocultured",
    "scaling_out":                  "scaling",
    "ccc_tool_out":                 "ccc_comparison",
}
STRAY_SUFFIXES = (".svg", ".png")
LEGACY_FIG_DIR = "legacy_figures"
LEGACY_FILE_DIR = "legacy_files"

# files that our newer work has superseded (moved to results/legacy_files/)
SUPERSEDED_FILES = [
    "lr_resource_dimitrov_consensus.csv",   # replaced by LR_databases/Human-merged-Jin-Dimitrov-LR-pairs.csv
]

# likely-redundant items — LISTED for you, never touched
REVIEW_PATHS = [
    "r_cellchat_bench",                       # duplicate of benchmark_cellchat_r
    "benchmark_cellchat_r/cellchat_test.R",   # old draft (superseded by cellchat_benchmark.R)
    "benchmark_cellchat_r/cellchat_test.txt",
    "log.txt",
    "benchmarks_ovarian_cancer.txt",
    "qxct_pred_classic_sim_oo.txt",
    "qxct_pred_sc_cocultured_monocultured_oo.txt",
]


def _move(src: Path, dst: Path, apply: bool):
    print(f"  MOVE  {src.relative_to(ROOT)}  ->  {dst.relative_to(ROOT)}")
    if apply:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            if src.is_dir() and dst.is_dir():
                for item in src.iterdir():
                    _move(item, dst / item.name, apply)
                if not any(src.iterdir()):
                    src.rmdir()
                return
            dst = dst.with_name(dst.stem + "_dup" + dst.suffix)
        shutil.move(str(src), str(dst))

ap = argparse.ArgumentParser()
ap.add_argument("--apply", action="store_true", help="actually move files (default: preview)")
ap.add_argument("--pycache", action="store_true", help="also delete __pycache__ dirs")
args = ap.parse_args()

mode = "APPLY" if args.apply else "DRY-RUN (nothing changed; use --apply)"
print(f"clean_outputs.py [{mode}]  root={ROOT}\n")

# 1) legacy result folders -> results/<category>/
print("Legacy output folders:")
hits = [(l, n) for l, n in FOLDER_MAP.items() if (ROOT / l).is_dir()]
for legacy, newname in hits:
    _move(ROOT / legacy, RESULTS / newname, args.apply)
if not hits:
    print("  (none)")

# 2) stray figures in python/ root -> results/legacy_figures/
print("\nStray figures in python/ root:")
strays = [p for p in ROOT.glob("*") if p.is_file() and p.suffix.lower() in STRAY_SUFFIXES]
for p in strays:
    _move(p, RESULTS / LEGACY_FIG_DIR / p.name, args.apply)
if not strays:
    print("  (none)")

# 3) superseded files -> results/legacy_files/
print("\nSuperseded files:")
sup = [ROOT / f for f in SUPERSEDED_FILES if (ROOT / f).exists()]
for p in sup:
    _move(p, RESULTS / LEGACY_FILE_DIR / p.name, args.apply)
if not sup:
    print("  (none)")

# 4) optional __pycache__ cleanup
if args.pycache:
    print("\n__pycache__ directories:")
    caches = list(ROOT.rglob("__pycache__"))
    for c in caches:
        print(f"  DELETE {c.relative_to(ROOT)}")
        if args.apply:
            shutil.rmtree(c, ignore_errors=True)
    if not caches:
        print("  (none)")

# 5) REVIEW MANUALLY — listed only, never touched
print("\nREVIEW MANUALLY (likely redundant — not moved or deleted):")
any_review = False
for rel in REVIEW_PATHS:
    p = ROOT / rel
    if p.exists():
        any_review = True
        kind = "dir " if p.is_dir() else "file"
        print(f"  [{kind}] {rel}")
if not any_review:
    print("  (none)")

print("\nDone." + ("" if args.apply else "  Re-run with --apply to perform the moves (review items are never auto-moved)."))
