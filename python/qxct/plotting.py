"""qxct.plotting — the single source of truth for QuantumXCT figures.

Every figure (the OO notebooks AND the analysis_tools panels) renders through the
functions here, so colours and layouts are identical everywhere.  These are
*pure rendering* helpers: they take already-computed distributions / contributions
and draw them.  How those numbers are obtained (real Aer shot sampling, or the
opt-in exact statevector for the SI verification) is the caller's decision — the
plotting code never re-scores anything.

Canonical palette (as in the manuscript):
    non-interacting / basal = steel blue
    QuantumXCT (model)      = medium purple
    interacting / target    = indigo
    cross-cell (LR) edges   = purple,   within-cell (GRN) edges = grey
    cell type 1 outline     = blue,     cell type 2 outline     = coral
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch

# --- palette ---------------------------------------------------------------
C_NON = "steelblue"       # non-interacting / basal / mono
C_QX = "mediumpurple"     # QuantumXCT model
C_INT = "indigo"          # interacting / co-culture target
C_LR = "#7E3FBF"          # cross-cell (ligand-receptor) edge
C_GRN = "#8C8C8C"         # within-cell (gene-network) edge
C_CT1 = "#2F6DB5"         # cell type 1 node outline / ellipse
C_CT2 = "#C9503C"         # cell type 2 node outline / ellipse
C_CT1_FILL = "#E6EFFA"
C_CT2_FILL = "#FBE9E5"
C_CT1_BOX = "#6F9FD8"     # circuit state-prep box, CT1
C_CT2_BOX = "#E08A78"     # circuit state-prep box, CT2
C_CRX = "#B79CED"         # CRX gate box


# --- small helpers ---------------------------------------------------------
def bitlabels(n):
    return [format(i, f"0{n}b") for i in range(2 ** n)]


def dist_to_vec(d, n=None):
    """A bitstring->value dict (or an array) -> dense length-2**n array."""
    if isinstance(d, (list, tuple, np.ndarray)):
        return np.asarray(d, float)
    d = d or {}
    if n is None:
        n = len(next(iter(d))) if d else 0
    return np.array([float(d.get(format(i, f"0{n}b"), 0.0)) for i in range(2 ** n)])


def sampled_marginals(psi, n1, n2, shots, rng):
    """N-shot measured marginals of the two registers = exactly what Aer returns
    for the final circuit (sample the joint |psi|^2, read c_measure1 / c_measure2).
    Used only where a caller wants a cheap Aer-equivalent draw; the notebooks use
    the real Aer measurement via qxct.analysis.post_optimization_distributions."""
    p = np.abs(np.asarray(psi)) ** 2
    p = p / p.sum()
    draws = rng.multinomial(int(shots), p).reshape(2 ** n2, 2 ** n1)   # row = reg2, col = reg1
    return draws.sum(axis=0) / shots, draws.sum(axis=1) / shots


# --- data joint histogram (raw counts, legacy legend) ----------------------
def data_histogram(ax, counts, n, title, color=C_NON, value_labels=True):
    """Frequency (raw counts) bar chart over all 2**n states, MSB-first."""
    y = dist_to_vec(counts, n)
    x = np.arange(2 ** n)
    ax.bar(x, y, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(bitlabels(n), rotation=90, fontsize=8 if n > 3 else 11)
    ax.set_xlabel(f"Bit String (q{n-1} ... q0)")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.margins(y=0.12)
    if value_labels:
        for xi, yi in zip(x, y):
            ax.annotate(f"{yi:g}", (xi, yi), textcoords="offset points", xytext=(0, 2),
                        ha="center", va="bottom", fontsize=7 if n > 3 else 9)


# --- state-occupancy comparison (mono vs QXCT vs co) -----------------------
def state_occupancy(ax, p_non, p_model, p_target, n, ct="CT", as_percent=True,
                    ylabel=True, title=None, labels=None):
    """3-way grouped bars: non-interacting / QuantumXCT / interacting.

    p_* may be dicts (bitstring->prob) or length-2**n arrays.  Values are shown
    as percentages by default (multiply by 100)."""
    vm = dist_to_vec(p_model, n); vt = dist_to_vec(p_target, n)
    s = 100.0 if as_percent else 1.0
    x = np.arange(2 ** n)
    lab = labels or (f"{ct}-Non-Interacting", f"{ct}-QXCT", f"{ct}-Interacting")
    if p_non is not None:                       # 3-way grouped bars
        v0 = dist_to_vec(p_non, n); w = 0.25
        ax.bar(x - w, s * v0, w, color=C_NON, label=lab[0])
        ax.bar(x,     s * vm, w, color=C_QX,  label=lab[1])
        ax.bar(x + w, s * vt, w, color=C_INT, label=lab[2])
    else:                                       # 2-way (model vs target)
        w = 0.4
        ax.bar(x - w / 2, s * vm, w, color=C_QX,  label=lab[1])
        ax.bar(x + w / 2, s * vt, w, color=C_INT, label=lab[2])
    ax.set_xticks(x); ax.set_xticklabels(bitlabels(n), rotation=60, ha="right", fontsize=9 if n > 3 else 11)
    if ylabel:
        ax.set_ylabel("Percentage (%)" if as_percent else "Probability", fontsize=13)
    ax.set_title(title or f"State Occupancy Comparison ({ct})")
    ax.yaxis.grid(True, ls=":", alpha=0.7); ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc="upper left")


# --- angle-optimization trace ----------------------------------------------
def angle_trace(ax, hist, title, color="tab:blue", label=None, best_so_far=True):
    # best_so_far=True plots the running minimum (monotone convergence curve). The raw
    # per-iteration objective is non-monotonic under shot-noisy COBYLA (it explores and
    # can spike up near the end); the best-so-far view matches the reported final angles.
    if hist is not None and len(hist):
        h = np.minimum.accumulate(hist) if best_so_far else hist
        ax.plot(range(len(h)), h, "-o", ms=5, lw=1.5, color=color, label=label)
    ax.set_xlabel("Optimization Iteration"); ax.set_ylabel("Combined KL Divergence"); ax.set_title(title)
    ax.grid(True)


# --- interaction network ----------------------------------------------------
def interaction_network(ax, topology, gene_list, n1, contributions=None,
                        name1="Cancer cells", name2="Fibroblasts (CAFs)",
                        bidirectional=True, strip_prefix="Gene_", caption=True):
    """Two-column network (one cell type per column) with the entangling edges.

    topology      : list of (control, target) gene-list indices.
    contributions : optional per-edge weight (list aligned with topology) that sets
                    arrow width, e.g. leave-one-out KL or ablation contribution.
                    None -> uniform width.
    bidirectional : draw double-headed arrows (the gate marks a relationship; the
                    direction is not asserted).
    """
    ng = len(gene_list)
    ax.set_xlim(0, 10); ax.set_ylim(-0.2, 6.6); ax.axis("off"); ax.set_aspect("equal")
    ax.add_patch(Ellipse((2.6, 3), 4.4, 5.4, color=C_CT1_FILL, ec=C_CT1))
    ax.add_patch(Ellipse((7.4, 3), 4.4, 5.4, color=C_CT2_FILL, ec=C_CT2))
    ax.text(2.6, 5.95, name1, ha="center", va="bottom", color=C_CT1, weight="bold")
    ax.text(7.4, 5.95, name2, ha="center", va="bottom", color=C_CT2, weight="bold")
    pos = {}
    for grp, cx, col in [(range(n1), 3.3, C_CT1), (range(n1, ng), 6.7, C_CT2)]:
        grp = list(grp); m = len(grp)
        ys = np.linspace(4.9, 1.1, m) if m > 1 else [3.0]
        for j, i in enumerate(grp):
            pos[i] = (cx, ys[j])
            ax.add_patch(plt.Circle(pos[i], 0.5, fc="white", ec=col, lw=2, zorder=3))
            name = gene_list[i].replace(strip_prefix, "G") if strip_prefix else gene_list[i]
            ax.text(*pos[i], name, ha="center", va="center", fontsize=6.5, weight="bold", zorder=4)
    w = list(contributions) if contributions is not None else [1.0] * len(topology)
    mx = max(w) if w and max(w) > 0 else 1.0
    style = "<|-|>" if bidirectional else "-|>"
    for (c, t), l in zip(topology, w):
        inter = (c < n1) != (t < n1)
        ax.add_patch(FancyArrowPatch(pos[c], pos[t], arrowstyle=style, mutation_scale=12,
                                     shrinkA=14, shrinkB=14,
                                     connectionstyle="arc3,rad=0.12" if inter else "arc3,rad=0.9",
                                     lw=1 + 5 * max(l, 0) / mx, color=C_LR if inter else C_GRN, zorder=2))
    if caption:
        note = "purple = cross-cell (LR), grey = within-cell (GRN); width = contribution"
        if bidirectional:
            note += "; arrows bidirectional (relationship, not asserted direction)"
        ax.text(5, 0.05, note, ha="center", fontsize=6.5, color="#555")
    return pos


# --- circuit diagram --------------------------------------------------------
def circuit_figure(topology, angles, gene_list, n1, fold=-1):
    """Return a matplotlib Figure of the circuit: two state-prep boxes (coloured
    per cell type) + the CRX topology + measurements."""
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    ng = len(gene_list)
    regs = [QuantumRegister(1, name=g) for g in gene_list]
    c1, c2 = ClassicalRegister(n1, "c_ct1"), ClassicalRegister(ng - n1, "c_ct2")
    qc = QuantumCircuit(*regs, c1, c2)
    qc.append(QuantumCircuit(n1, name="ψ CT1 (mono)").to_gate(), list(range(n1)))
    qc.append(QuantumCircuit(ng - n1, name="ψ CT2 (mono)").to_gate(), list(range(n1, ng)))
    qc.barrier()
    for (c, t), th in zip(topology, angles):
        qc.crx(round(float(th), 2), c, t)
    qc.barrier()
    qc.measure(list(range(n1)), c1); qc.measure(list(range(n1, ng)), c2)
    return qc.draw("mpl", fold=fold, style={"displaycolor": {
        "crx": (C_CRX, "#000000"),
        "ψ CT1 (mono)": (C_CT1_BOX, "#000000"),
        "ψ CT2 (mono)": (C_CT2_BOX, "#000000")}})
