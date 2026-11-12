# analysis_tools/compile_budget_table.py
# Compile the D=1 / D=2 capacity-reference comparison (QuantumXCT vs TN transform,
# Born machine and optimal transport) from the benchB sweep into one spreadsheet
# and one LaTeX table for the SI. Reads the exact-scoring benchB outputs written by
# run_scopes_and_benchB.py; no re-fitting here, just tabulation.
#
# Inputs  : results/analyses/benchA_benchB_exact/{ovarian,synthetic}_benchB_sweep.csv
# Outputs : results/analyses/budget_capacity_table.xlsx   (spreadsheet)
#           results/analyses/budget_capacity_table.tex    (SI table, \input-able)
import os
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side

R = os.environ.get("QXCT_RESULTS", "results/analyses")
BENCH = os.path.join(R, "benchA_benchB_exact")
DATASETS = [("ovarian", "Ovarian"), ("synthetic", "Synthetic")]

# canonical row order and display labels for the methods / settings
ORDER = [
    ("QuantumXCT", None),
    ("TN transform", "bond_dim=1"),
    ("TN transform", "bond_dim=2"),
    ("Born machine", "bond_dim=1"),
    ("Born machine", "bond_dim=2"),
    ("Optimal Transport", None),
]
LABEL = {
    ("QuantumXCT", None): "QuantumXCT (best effort)",
    ("TN transform", "bond_dim=1"): "TN transform, D=1 (product)",
    ("TN transform", "bond_dim=2"): "TN transform, D=2 (pairwise)",
    ("Born machine", "bond_dim=1"): "Born machine, D=1 (product)",
    ("Born machine", "bond_dim=2"): "Born machine, D=2 (pairwise)",
    ("Optimal Transport", None): "Optimal transport (top-m)",
}


def _fmt_kl(x):
    x = float(x)
    return "<1e-4" if x < 1e-4 else f"{x:.4f}"


# ---- optional Local-RY null (per-gene rotation; no interactions) -----------
# KL comes from the exact summary; parameter count is one RY angle per qubit.
import json
_LOCAL_RY = {}
try:
    _e = json.load(open(os.path.join(R, "exact_summary.json")))
    from analysis_tools.data import OV_CT1, OV_CT2, SIM_CT1, SIM_CT2
    _NGENES = {"ovarian": len(OV_CT1) + len(OV_CT2), "synthetic": len(SIM_CT1) + len(SIM_CT2)}
    for key in ("ovarian", "synthetic"):
        if key in _e and "local_RY_only_kl" in _e[key]:
            _LOCAL_RY[key] = (float(_e[key]["local_RY_only_kl"]), _NGENES.get(key))
except Exception as _ex:
    print(f"[compile_budget_table] Local-RY null not added ({_ex}); run the exact step first.")

# ---- assemble a tidy long table across both datasets -----------------------
records = []
for key, nice in DATASETS:
    df = pd.read_csv(os.path.join(BENCH, f"{key}_benchB_sweep.csv"))
    for method, setting in ORDER:
        if setting is None:
            row = df[df["Method"] == method]
        else:
            row = df[(df["Method"] == method) & (df["setting"] == setting)]
        if not len(row):
            continue
        r = row.iloc[0]
        records.append({
            "Dataset": nice,
            "Method": LABEL[(method, setting)],
            "setting": r["setting"],
            "KL (c1+c2)": _fmt_kl(r["KL (c1+c2)"]),
            "Stored params": int(r["# params"]),
            "Cond. on basal": r["conditioned on basal"],
            "Interpretable net": r["interpretable network"],
        })
    if key in _LOCAL_RY:
        kl_ry, n_ry = _LOCAL_RY[key]
        records.append({
            "Dataset": nice,
            "Method": "Local-RY (per-gene null)",
            "setting": "1 RY/qubit",
            "KL (c1+c2)": _fmt_kl(kl_ry),
            "Stored params": int(n_ry) if n_ry is not None else "",
            "Cond. on basal": "yes",
            "Interpretable net": "no",
        })
tidy = pd.DataFrame.from_records(records)
os.makedirs(R, exist_ok=True)
tidy.to_csv(os.path.join(R, "budget_capacity_table.csv"), index=False)

# ---- spreadsheet -----------------------------------------------------------
wb = Workbook()
ws = wb.active
ws.title = "Budget capacity (D1 vs D2)"
HEAD = Font(name="Arial", bold=True, color="FFFFFF")
BODY = Font(name="Arial")
HFILL = PatternFill("solid", fgColor="4472C4")
QXFILL = PatternFill("solid", fgColor="E7EEF9")
thin = Side(style="thin", color="D0D0D0")
BORD = Border(left=thin, right=thin, top=thin, bottom=thin)
cols = ["Dataset", "Method", "KL (c1+c2)", "Stored params",
        "Cond. on basal", "Interpretable net"]
ws.append(cols)
for c in range(1, len(cols) + 1):
    cell = ws.cell(row=1, column=c)
    cell.font = HEAD; cell.fill = HFILL; cell.border = BORD
    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
for _, rr in tidy.iterrows():
    ws.append([rr[c] for c in cols])
    r = ws.max_row
    is_qx = str(rr["Method"]).startswith("QuantumXCT")
    for c in range(1, len(cols) + 1):
        cell = ws.cell(row=r, column=c)
        cell.font = Font(name="Arial", bold=is_qx); cell.border = BORD
        cell.alignment = Alignment(horizontal="center" if c >= 3 else "left", vertical="center")
        if is_qx:
            cell.fill = QXFILL
widths = [12, 30, 12, 14, 15, 17]
for i, w in enumerate(widths, 1):
    ws.column_dimensions[chr(64 + i)].width = w
ws.freeze_panes = "A2"
# legend / provenance so the sheet is self-documenting
note_r = ws.max_row + 2
ws.cell(row=note_r, column=1, value=(
    "D is the tensor-network bond dimension: D=1 is a product state (independent genes), "
    "D=2 captures pairwise correlations. 'Stored params' are raw tensor/plan entries (capacity "
    "reference), not the gauge-free effective degrees of freedom used in the budget-matched "
    "table. All KL values use exact (noise-free) output probabilities. "
    "Source: results/analyses/benchA_benchB_exact/{ovarian,synthetic}_benchB_sweep.csv")).font = Font(
        name="Arial", italic=True, size=9)
ws.merge_cells(start_row=note_r, start_column=1, end_row=note_r, end_column=6)
ws.cell(row=note_r, column=1).alignment = Alignment(wrap_text=True, vertical="top")
wb.save(os.path.join(R, "budget_capacity_table.xlsx"))

# ---- LaTeX table for the SI ------------------------------------------------
def _tex_rows(nice):
    sub = tidy[tidy["Dataset"] == nice]
    out = []
    for _, rr in sub.iterrows():
        m = rr["Method"].replace("D=1", "$D{=}1$").replace("D=2", "$D{=}2$")
        bold = str(rr["Method"]).startswith("QuantumXCT")
        kl = rr["KL (c1+c2)"].replace("<1e-4", "$<10^{-4}$")
        cond = rr["Cond. on basal"].replace("*", "")
        if str(rr["Method"]).startswith("Born"):
            cond = cond + "$^{*}$"
        cells = [m, kl, str(rr["Stored params"]), cond, rr["Interpretable net"]]
        line = " & ".join(("\\textbf{" + c + "}" if bold and i == 0 else c) for i, c in enumerate(cells))
        out.append("    " + line + "\\\\")
    return "\n".join(out)

tex = r"""\begin{table}[h]
\centering
\caption{Capacity-reference comparison (exact scoring). For the tensor-network baselines $D$ is
the bond dimension: $D{=}1$ is a product state (independent genes) and $D{=}2$ captures pairwise
correlations. ``Params'' are stored tensor/plan entries (a capacity reference, not gauge-free
effective degrees of freedom); ``Cond.''~= conditioned on the basal
state; ``Net.''~= returns an interaction network. KL is the summed per-register
$D_{\mathrm{KL}}(P_{\mathrm{model}}\|Q_{\mathrm{co}})$ on exact output probabilities.}
\label{tab:si_capacity}
\begin{tabular}{lrrcc}
\toprule
Method & KL (c1+c2) & Params & Cond. & Net.\\
\midrule
\multicolumn{5}{l}{\textit{Ovarian cancer--fibroblast}}\\
__OVARIAN__
\midrule
\multicolumn{5}{l}{\textit{Synthetic benchmark}}\\
__SYNTH__
\bottomrule
\end{tabular}
\par\smallskip
{\footnotesize $^{*}$The Born machine is not conditioned on the basal state and serves as an oracle reference.}
\end{table}
"""
tex = tex.replace("__OVARIAN__", _tex_rows("Ovarian")).replace("__SYNTH__", _tex_rows("Synthetic"))
open(os.path.join(R, "budget_capacity_table.tex"), "w").write(tex)

print("wrote:")
for f in ("budget_capacity_table.csv", "budget_capacity_table.xlsx", "budget_capacity_table.tex"):
    print("  ", os.path.join(R, f))
print()
print(tidy.to_string(index=False))
