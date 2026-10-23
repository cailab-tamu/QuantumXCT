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
                    ylabel=True, title=None, labels=None,
                    tick_fontsize=None, ylabel_fontsize=13, title_fontsize=None,
                    legend_fontsize=9):
    """3-way grouped bars: non-interacting / QuantumXCT / interacting.

    p_* may be dicts (bitstring->prob) or length-2**n arrays.  Values are shown
    as percentages by default (multiply by 100).

    Font knobs (``tick_fontsize`` for the bitstring x labels, ``title_fontsize``,
    ``legend_fontsize``, ``ylabel_fontsize``) default to the historical sizes; pass
    larger values for poster-scale panels."""
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
    _tick_fs = tick_fontsize if tick_fontsize is not None else (9 if n > 3 else 11)
    ax.set_xticks(x); ax.set_xticklabels(bitlabels(n), rotation=60, ha="right", fontsize=_tick_fs)
    if ylabel:
        ax.set_ylabel("Percentage (%)" if as_percent else "Probability", fontsize=ylabel_fontsize)
    ax.set_title(title or f"State Occupancy Comparison ({ct})",
                 fontsize=title_fontsize if title_fontsize is not None else None)
    ax.yaxis.grid(True, ls=":", alpha=0.7); ax.set_axisbelow(True)
    ax.legend(fontsize=legend_fontsize, loc="upper left")


# --- angle-optimization trace ----------------------------------------------
def angle_trace(ax, hist, title, color="tab:blue", label=None, best_so_far=False):
    # Default: plot the RAW per-evaluation objective. COBYLA (a gradient-free trust-region
    # method) deliberately samples points away from the current best, so the trace bumps up
    # and down as it explores and escapes flat/vanishing-gradient regions; that exploration
    # is the point and should be visible. best_so_far=True instead plots the running minimum.
    if hist is not None and len(hist):
        h = np.minimum.accumulate(hist) if best_so_far else hist
        ax.plot(range(len(h)), h, "-o", ms=5, lw=1.5, color=color, label=label)
    ax.set_xlabel("Optimization Iteration"); ax.set_ylabel("Combined KL Divergence"); ax.set_title(title)
    ax.grid(True)


# --- interaction network ----------------------------------------------------
def _lr_edge_set(lr_pairs):
    """Turn a list of (ligand, receptor) gene-name tuples into a set of
    unordered, upper-cased frozenset keys for edge lookup (direction not asserted)."""
    s = set()
    for a, b in (lr_pairs or []):
        s.add(frozenset((str(a).upper(), str(b).upper())))
    return s


def redirect_through_hub(topology, gene_list, n1, lr_pairs, contributions=None):
    """Re-route non-ligand-receptor cross-cell links through the validated LR hub.

    Take-home: the only real cross-cell links are known ligand-receptor pairs. A
    predicted cross-cell link that is not itself an LR pair is re-interpreted by
    sending the RECEPTOR to its true, known ligand (the hub), because a receptor is
    only physically activated by its cognate ligand. The predicted source is handled
    by what it is:

      * source is an ordinary gene / transcription factor (not a ligand or receptor
        of any pair): it is chained to the receptor's ligand within its own cell,
        giving source -> ligand <-> receptor. Example: STAT3 (cancer) -> PDGFRB (CAF)
        is not an LR pair, so it becomes STAT3 -> PDGFB -> PDGFRB (STAT3 acts on the
        ligand PDGFB, which signals to PDGFRB).
      * source is itself a ligand or receptor (a hub gene): it is NOT chained to a
        second ligand (that would be a meaningless ligand->ligand edge); the link
        simply resolves to the receptor's own hub. Example: PDGFB (a ligand) ->
        TGFBR2 becomes just the hub TGFB1 <-> TGFBR2 (TGFBR2's real ligand), and the
        PDGFB source is dropped. PDGFB keeps its own hub from its own LR link.

    Crucially this NEVER invents a hub: a cross-cell link is drawn as a ligand-receptor
    hub only if that validated pair was actually predicted by the search (present in
    ``topology``). Each kept edge carries its predicted KL contribution as the arrow
    width, so a near-zero prediction stays near-zero, not promoted to a strong hub.
    A spurious ligand -> wrong-receptor link (e.g. PDGFB -> TGFBR2, which is not a
    pair and whose true hub TGFB1 <-> TGFBR2 was not predicted) is dropped, not
    reinvented. Within-cell and validated-and-predicted LR links are kept.

    Returns ``(new_topology, kinds, new_contributions)`` with ``kinds[i]`` one of
    'grn' (within-cell), 'hub' (validated LR that was predicted), or 'rerouted'
    (an ordinary source gene moved onto a receptor's ligand, within its cell).
    """
    name = lambda i: str(gene_list[i]).upper()
    cell = lambda i: 0 if i < n1 else 1
    idx_of = {}
    for i, g in enumerate(gene_list):
        idx_of.setdefault(str(g).upper(), i)
    lr_set = _lr_edge_set(lr_pairs)
    # directional: each receptor -> its known ligand(s); hub genes = all ligands + receptors
    receptor_to_ligands, hub_genes = {}, set()
    for lig, rec in (lr_pairs or []):
        L, R = str(lig).upper(), str(rec).upper()
        receptor_to_ligands.setdefault(R, []).append(L)
        hub_genes.add(L); hub_genes.add(R)

    w = list(contributions) if contributions is not None else [1.0] * len(topology)
    out, kinds, outw = [], [], []
    seen = {}

    def _add(edge, kind, weight):
        key = frozenset(edge)
        if key in seen:                              # keep the stronger style/weight
            j = seen[key]
            outw[j] = max(outw[j], weight)
            if kind == "hub":
                kinds[j] = "hub"
            return
        seen[key] = len(out)
        out.append(edge); kinds.append(kind); outw.append(weight)

    for (c, t), wt in zip(topology, w):
        inter = cell(c) != cell(t)
        if not inter:
            _add((c, t), "grn", wt); continue
        if frozenset((name(c), name(t))) in lr_set:
            _add((c, t), "hub", wt); continue        # validated LR that WAS predicted
        # non-LR cross-cell: move an ordinary source gene onto a known receptor's ligand.
        # Never add a hub here (that would invent a non-predicted pair); drop otherwise.
        for rcp, src in ((c, t), (t, c)):
            if name(rcp) not in receptor_to_ligands:
                continue                             # this endpoint is not a known receptor
            if name(src) in hub_genes:
                break                                # ligand/receptor source -> spurious, drop
            for lig in receptor_to_ligands[name(rcp)]:
                li = idx_of.get(lig)
                if li is not None and li != src and cell(li) == cell(src):
                    _add((src, li), "rerouted", wt)  # source gene -> the receptor's ligand (within cell)
                    break
            break
        # cross-cell links with no known-receptor endpoint, or a ligand/receptor source,
        # are dropped from the redirected view (they are exploratory candidates only).
    return out, kinds, outw


def interaction_network(ax, topology, gene_list, n1, contributions=None,
                        name1="Cancer cells", name2="Fibroblasts (CAFs)",
                        bidirectional=True, strip_prefix="Gene_", caption=True,
                        lr_pairs=None, lr_only=False,
                        label_fontsize=6.5, name_fontsize=None, caption_fontsize=None):
    """Two-column network (one cell type per column) with the entangling edges.

    topology      : list of (control, target) gene-list indices.
    contributions : optional per-edge weight (list aligned with topology) that sets
                    arrow width, e.g. leave-one-out KL or ablation contribution.
                    None -> uniform width.
    bidirectional : draw double-headed arrows (the gate marks a relationship; the
                    direction is not asserted).
    lr_pairs      : optional list of (ligand, receptor) gene-name tuples (e.g. from
                    ``curate_lr_pairs``). When given, a cross-cell edge whose two
                    genes form one of these pairs is drawn as a VALIDATED ligand-
                    receptor link (solid purple); any other cross-cell edge is a
                    candidate link (dashed, lighter). This restores the LR linking:
                    the database decides which discovered hubs are known biology.
    lr_only       : when True (the "restricted" view), draw only the validated LR
                    cross-cell edges and drop the unvalidated cross-cell candidates.
                    Within-cell (GRN) edges are always kept. The discovered topology
                    is unchanged; this only filters what is shown.
    """
    # font sizes: ``label_fontsize`` sets the gene-name node labels (default 6.5,
    # the historical value); ``name_fontsize`` the two cell-type column titles
    # (defaults to label_fontsize+2); ``caption_fontsize`` the legend line
    # (defaults to label_fontsize). Pass larger values for poster-scale figures.
    name_fontsize = name_fontsize if name_fontsize is not None else label_fontsize + 2
    caption_fontsize = caption_fontsize if caption_fontsize is not None else label_fontsize
    ng = len(gene_list)
    lr_set = _lr_edge_set(lr_pairs)
    ax.set_xlim(0, 10); ax.set_ylim(-0.2, 6.6); ax.axis("off"); ax.set_aspect("equal")
    ax.add_patch(Ellipse((2.6, 3), 4.4, 5.4, color=C_CT1_FILL, ec=C_CT1))
    ax.add_patch(Ellipse((7.4, 3), 4.4, 5.4, color=C_CT2_FILL, ec=C_CT2))
    ax.text(2.6, 5.95, name1, ha="center", va="bottom", color=C_CT1, weight="bold", fontsize=name_fontsize)
    ax.text(7.4, 5.95, name2, ha="center", va="bottom", color=C_CT2, weight="bold", fontsize=name_fontsize)
    pos = {}
    for grp, cx, col in [(range(n1), 3.3, C_CT1), (range(n1, ng), 6.7, C_CT2)]:
        grp = list(grp); m = len(grp)
        ys = np.linspace(4.9, 1.1, m) if m > 1 else [3.0]
        for j, i in enumerate(grp):
            pos[i] = (cx, ys[j])
            ax.add_patch(plt.Circle(pos[i], 0.5, fc="white", ec=col, lw=2, zorder=3))
            name = gene_list[i].replace(strip_prefix, "G") if strip_prefix else gene_list[i]
            ax.text(*pos[i], name, ha="center", va="center", fontsize=label_fontsize, weight="bold", zorder=4)
    w = list(contributions) if contributions is not None else [1.0] * len(topology)
    mx = max(w) if w and max(w) > 0 else 1.0
    style = "<|-|>" if bidirectional else "-|>"
    have_lr = bool(lr_set)
    for (c, t), l in zip(topology, w):
        inter = (c < n1) != (t < n1)
        is_lr = inter and have_lr and frozenset(
            (str(gene_list[c]).upper(), str(gene_list[t]).upper())) in lr_set
        if lr_only and inter and not is_lr:
            continue  # restricted view: hide unvalidated cross-cell candidates
        if not inter:
            color, ls = C_GRN, "solid"                       # within-cell (GRN)
        elif not have_lr:
            color, ls = C_LR, "solid"                        # legacy: all cross-cell as LR
        elif is_lr:
            color, ls = C_LR, "solid"                        # validated ligand-receptor
        else:
            color, ls = "#B9A7D9", (0, (4, 2))               # candidate cross-cell (dashed)
        ax.add_patch(FancyArrowPatch(pos[c], pos[t], arrowstyle=style, mutation_scale=12,
                                     shrinkA=14, shrinkB=14, linestyle=ls,
                                     connectionstyle="arc3,rad=0.12" if inter else "arc3,rad=0.9",
                                     lw=1 + 5 * max(l, 0) / mx, color=color, zorder=2))
    if caption:
        if have_lr:
            note = ("solid purple = validated ligand-receptor, dashed = candidate cross-cell, "
                    "grey = within-cell (GRN); width = contribution")
            if lr_only:
                note = ("solid purple = validated ligand-receptor, grey = within-cell (GRN); "
                        "width = contribution (candidate cross-cell links hidden)")
        else:
            note = "purple = cross-cell (LR), grey = within-cell (GRN); width = contribution"
        if bidirectional:
            note += "; arrows bidirectional (relationship, not asserted direction)"
        # wrap the legend onto two lines: colour key on line 1, the width/arrows
        # note on line 2, so it stays readable when the caption font is large.
        note = note.replace("; width = contribution", "\nwidth = contribution")
        ax.text(5, -0.05, note, ha="center", va="top", fontsize=caption_fontsize,
                color="#555", linespacing=1.4)
    return pos


# --- circuit diagram --------------------------------------------------------
def circuit_figure(topology, angles, gene_list, n1, fold=-1, fontsize=None):
    """Return a matplotlib Figure of the circuit: two state-prep boxes (coloured
    per cell type) + the CRX topology + measurements.

    ``fontsize`` (gate/label text) defaults to qiskit's own; pass a slightly
    larger value for poster-scale circuits (register/gate labels render bolder
    because qiskit's mpl drawer draws them heavier at larger sizes)."""
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
    style = {"displaycolor": {
        "crx": (C_CRX, "#000000"),
        "ψ CT1 (mono)": (C_CT1_BOX, "#000000"),
        "ψ CT2 (mono)": (C_CT2_BOX, "#000000")}}
    if fontsize is not None:
        style["fontsize"] = fontsize            # gate-label text
        style["subfontsize"] = max(fontsize - 2, 8)  # parameter / sub text
    return qc.draw("mpl", fold=fold, style=style)
