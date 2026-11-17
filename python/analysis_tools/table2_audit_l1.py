"""Recompute the L1 ovarian gate-contribution + LOO tables for BOTH algorithms
with exact probabilities, to mirror Table 3 (L2) and let SI Table S4 report
L1 and L2 side by side. Twin of table2_audit.py (which does L2).

Topologies are the saved L1 'all'-scope results from
results/analyses/ovarian_exact_scopes_algorithms.csv (norm=l1): no new search is
run here; angles are exactly re-optimized and the per-gate sequential/isolated/
leave-one-out contributions are computed on exact state-vector probabilities.

Run from QuantumXCT/python:
    python analysis_tools/table2_audit_l1.py
Writes: results/analyses/table2_exact_audit_L1.csv
"""
import os, sys; sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
from qxct import vector_normalize_dictionary_values as vn
from qxct.exact import ExactScorer, exact_contribution_table
from analysis_tools.data import load_ovarian, OV_CT1, OV_CT2

g = OV_CT1 + OV_CT2
ix = {x: i for i, x in enumerate(g)}
H = load_ovarian()
sc = ExactScorer(*[vn(H[k], "l1") for k in ("ct1_mo", "ct2_mo", "ct1_co", "ct2_co")])

# L1 'all'-scope topologies (search order preserved) from
# results/analyses/ovarian_exact_scopes_algorithms.csv
E = {
    "N-wise": [("STAT3", "PDGFRB"), ("PDGFRB", "PDGFB"), ("PDGFB", "PDGFRB"),
               ("TGFBR1", "PDGFB"), ("TGFBR1", "TGFB1"), ("TGFB1", "TGFBR2")],
    "Multi-epoch": [("STAT3", "PDGFRB"), ("PDGFRB", "PDGFB"),
                    ("PDGFB", "TGFBR2"), ("TGFBR1", "TGFB1")],
}

out = []
for alg, edges in E.items():
    topo = [(ix[a], ix[b]) for a, b in edges]
    angles = sc.optimize_angles(topo, restarts=10)[0]          # exact re-optimized
    df, meta = exact_contribution_table(sc, topo, angles, g, nshots=5000, n_rep=300)
    df.insert(0, "angles", "exact re-optimized")
    df.insert(0, "algorithm", alg)
    for k in ("baseline_kl", "final_kl", "total_reduction", "pct_reduction", "shot_sd_final_kl"):
        df[k] = meta[k]
    out.append(df)
    print(alg, {k: round(v, 4) for k, v in meta.items()})
    print(df[["gate", "theta_rad", "KL_cumulative", "dKL_step",
              "pct_of_total_reduction_step", "isolated_reduction",
              "LOO_increase", "pct_LOO_of_total_reduction"]].round(4).to_string())

pd.concat(out).to_csv("results/analyses/table2_exact_audit_L1.csv", index=False)
print("\nwrote results/analyses/table2_exact_audit_L1.csv")
