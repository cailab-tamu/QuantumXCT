"""Data joint histograms (raw counts) in the legacy legend style, for BOTH datasets.

Reproduces the "Data Joint Histogram" panels: y = Frequency (raw cell counts),
x = every bit string in MSB-first order labelled "Bit String (q{n-1} ... q0)",
value labels on each bar, titles "Data Joint Histogram {Non-Interacting|
Interacting}-CT{1|2}".  Uses the same load_sim / load_ovarian pipeline as the
rest of the review tools, so the counts match the states the circuits are
encoded from.

Outputs (results/analyses/figures) and mirrored into the dataset folders:
    {synthetic,ovarian}_data_hist_{noninteracting,interacting}_CT{1,2}.png/.pdf
    {synthetic,ovarian}_data_hist_grid.png/.pdf        (2x2: conditions x cell types)
"""
import os, sys, shutil
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analysis_tools.data import load_ovarian, load_sim, OV_CT1, OV_CT2, SIM_CT1, SIM_CT2
from qxct.plotting import data_histogram   # single source of truth

OUT = "results/analyses/figures"; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.titleweight": "bold", "axes.labelweight": "bold"})

DATASETS = [
    ("synthetic", load_sim, SIM_CT1, SIM_CT2, "results/classic_sim"),
    ("ovarian",  load_ovarian, OV_CT1, OV_CT2, "results/ovarian_cocultured"),
]


def counts_vec(d, n):
    return [int(d.get(format(i, f"0{n}b"), 0)) for i in range(2 ** n)]


def draw(ax, d, n, title):
    data_histogram(ax, {k: int(v) for k, v in d.items()}, n, title)


def save(fig, path):
    fig.savefig(path + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(path + ".pdf", bbox_inches="tight")
    fig.savefig(path + ".svg", bbox_inches="tight")
    plt.close(fig)

for name, loader, ct1, ct2, mirror in DATASETS:
    H = loader(); n1, n2 = len(ct1), len(ct2)
    panels = [
        ("noninteracting", "CT1", H["ct1_mo"], n1, "Data Joint Histogram Non-Interacting-CT1"),
        ("noninteracting", "CT2", H["ct2_mo"], n2, "Data Joint Histogram Non-Interacting-CT2"),
        ("interacting",   "CT1", H["ct1_co"], n1, "Data Joint Histogram Interacting-CT1"),
        ("interacting",   "CT2", H["ct2_co"], n2, "Data Joint Histogram Interacting-CT2"),
    ]
    for cond, ct, d, n, title in panels:
        fig, ax = plt.subplots(figsize=(5.2 if n <= 2 else 7.5, 4.2))
        draw(ax, d, n, title)
        fig.tight_layout()
        base = os.path.join(OUT, f"{name}_data_hist_{cond}_{ct}")
        save(fig, base)
        if os.path.isdir(mirror):
            shutil.copy(base + ".png", os.path.join(mirror, os.path.basename(base) + ".png"))
    # 2x2 grid: rows = conditions, cols = cell types
    fig, axs = plt.subplots(2, 2, figsize=(13, 8),
                            gridspec_kw={"width_ratios": [1, 1.35]})
    draw(axs[0, 0], H["ct1_mo"], n1, "Data Joint Histogram Non-Interacting-CT1")
    draw(axs[0, 1], H["ct2_mo"], n2, "Data Joint Histogram Non-Interacting-CT2")
    draw(axs[1, 0], H["ct1_co"], n1, "Data Joint Histogram Interacting-CT1")
    draw(axs[1, 1], H["ct2_co"], n2, "Data Joint Histogram Interacting-CT2")
    fig.tight_layout()
    base = os.path.join(OUT, f"{name}_data_hist_grid")
    save(fig, base)
    if os.path.isdir(mirror):
        shutil.copy(base + ".png", os.path.join(mirror, os.path.basename(base) + ".png"))
    print(f"{name}: wrote 4 panels + grid  (CT1 {n1}q, CT2 {n2}q)", flush=True)
