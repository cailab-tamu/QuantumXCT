# collector_ccc_comparison.py
# Assemble ONE comparison across CellChat / LIANA / scTenifoldXct / QuantumXCT and
# save a table + plots to results/ccc_comparison/.
# ENV: qiskit-env is fine (only needs pandas/numpy/matplotlib); reads the CSVs each
# tool wrote. Run AFTER the three tool scripts + the ovarian benchmark notebook.
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))          # .../QuantumXCT/python
OUT  = os.path.join(HERE, "results", "ccc_comparison")
QX_ISO = os.path.join(HERE, "results", "ovarian_cocultured", "inter_isolated_contribution.csv")
os.makedirs(OUT, exist_ok=True)

VALID_LR = [('PDGFB', 'PDGFRB'), ('TGFB1', 'TGFBR1'), ('TGFB1', 'TGFBR2')]


def _safe_read(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def _first(*paths):
    """First existing CSV. Avoids ``df_a or df_b``, which raises
    'truth value of a DataFrame is ambiguous' whenever df_a exists."""
    for p in paths:
        df = _safe_read(p)
        if df is not None:
            return df
    return None


def _has_pair(df, lig, rec, lig_col, rec_col):
    if df is None or lig_col not in df.columns or rec_col not in df.columns:
        return None
    hit = df[(df[lig_col].astype(str).str.contains(lig, case=False, na=False)) &
             (df[rec_col].astype(str).str.contains(rec, case=False, na=False))]
    return hit if len(hit) else None


def _qx_iso_df():
    """Load QuantumXCT LR-scope isolated-contribution table if present."""
    df = _safe_read(QX_ISO)
    return df


def collect(qx_isolated_df=None):
    # prefer the CellChat DIFFERENTIAL (up-in-Co) result; fall back to per-condition Co
    cellchat = _first(os.path.join(OUT, "cellchat_co_vs_mo_up.csv"),
                   os.path.join(OUT, "cellchat_Co_all.csv"))
    # prefer LIANA+ df_to_lr differential (interaction_stat); fall back to heuristic / per-condition
    liana_co = _first(os.path.join(OUT, "liana_co_vs_mo_diff.csv"),
                   os.path.join(OUT, "liana_Co_all.csv"))
    xct      = _first(os.path.join(OUT, "sctenifoldxct_co_vs_mo_diff.csv"),
                   os.path.join(OUT, "sctenifoldxct_co_all.csv"))
    if qx_isolated_df is None:
        qx_isolated_df = _qx_iso_df()

    rows = []
    for lig, rec in VALID_LR:
        row = {"Ligand": lig, "Receptor": rec}

        cc = _has_pair(cellchat, lig, rec, "ligand", "receptor")
        row["CellChat prob"] = (round(float(cc["prob"].max()), 4)
                                if (cc is not None and "prob" in cc.columns) else np.nan)

        lh = _has_pair(liana_co, lig, rec, "ligand_complex", "receptor_complex")
        if lh is None:
            row["LIANA dStrength (Co-Mo)"] = np.nan
        elif "interaction_stat" in lh.columns:                         # LIANA+ df_to_lr differential
            row["LIANA dStrength (Co-Mo)"] = round(float(lh["interaction_stat"].max()), 4)
        elif "delta_magnitude_strength" in lh.columns:                 # older heuristic contrast
            row["LIANA dStrength (Co-Mo)"] = round(float(lh["delta_magnitude_strength"].max()), 4)
        elif "magnitude_rank" in lh.columns:                           # per-condition fallback
            row["LIANA dStrength (Co-Mo)"] = round(float(1.0 - lh["magnitude_rank"].min()), 4)
        else:
            row["LIANA dStrength (Co-Mo)"] = np.nan

        if xct is not None:
            mask = xct.apply(lambda r: lig in str(r.values) and rec in str(r.values), axis=1)
            row["scTenifoldXct hit"] = "yes" if mask.any() else "no"
        else:
            row["scTenifoldXct hit"] = np.nan

        row["QuantumXCT KL contrib"] = np.nan
        if qx_isolated_df is not None and "Gate" in qx_isolated_df.columns:
            g = qx_isolated_df[qx_isolated_df["Gate"].astype(str).str.contains(
                f"{lig}.*{rec}|{rec}.*{lig}", case=False, regex=True, na=False)]
            if len(g):
                row["QuantumXCT KL contrib"] = round(float(g["Marginal Contribution"].max()), 4)

        rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(OUT, "ccc_comparison_table.csv"), index=False)
    return table


def plot(table):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{l}->{r}" for l, r in zip(table["Ligand"], table["Receptor"])]

    # (1) per-tool score panels
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col, ttl in [
        (axes[0], "CellChat prob", "CellChat probability (higher=stronger)"),
        (axes[1], "LIANA dStrength (Co-Mo)", "LIANA Co-Mo strength (higher=stronger in Co)"),
        (axes[2], "QuantumXCT KL contrib", "QuantumXCT KL contribution"),
    ]:
        vals = table[col].fillna(0) if col in table else [0] * len(labels)
        ax.bar(labels, vals, color="#56B4E9", alpha=0.85)
        ax.set_title(ttl); ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ccc_comparison_panels.png"), dpi=140)
    plt.close(fig)

    # (2) detection heatmap: 1 = pair reported/nonzero by that tool
    tools = ["CellChat prob", "LIANA dStrength (Co-Mo)", "scTenifoldXct hit", "QuantumXCT KL contrib"]
    M = np.zeros((len(labels), len(tools)))
    for i, (_, r) in enumerate(table.iterrows()):
        for j, t in enumerate(tools):
            v = r.get(t)
            if t == "scTenifoldXct hit":
                M[i, j] = 1.0 if str(v).lower() == "yes" else 0.0
            elif t == "LIANA dStrength (Co-Mo)":
                M[i, j] = 1.0 if pd.notna(v) else 0.0           # present in contrast => detected
            else:
                M[i, j] = 1.0 if (pd.notna(v) and float(v) != 0.0) else 0.0
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(tools))); ax.set_xticklabels(tools, rotation=30, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_title("LR-pair detection across methods (blue = detected)")
    for i in range(len(labels)):
        for j in range(len(tools)):
            ax.text(j, i, "✓" if M[i, j] else "—", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ccc_comparison_heatmap.png"), dpi=140)
    plt.close(fig)
    print(f"Saved plots to {OUT}")

tbl = collect()
print(tbl.to_string(index=False))
plot(tbl)
print(f"\nComparison table -> {os.path.join(OUT, 'ccc_comparison_table.csv')}")
