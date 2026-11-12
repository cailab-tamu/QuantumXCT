"""Re-run the author's Benchmark A (compare_scopes: GRN+LR / GRN / LR, with cumulative,
isolated and leave-one-out tables) and Benchmark B (benchmark_budget_matched, best-effort
QuantumXCT, TN/Born at D=1 and D=2) with exact scoring."""
import os, sys, io, contextlib, json
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
from qxct import RunConfig, EncodingConfig, BackendConfig, SearchConfig, OptimizerConfig
from qxct.exact import use_exact_scoring
from qxct.analysis import compare_scopes
from qxct.baselines import benchmark_budget_matched
from analysis_tools.data import load_ovarian, load_sim, OV_CT1, OV_CT2, SIM_CT1, SIM_CT2
use_exact_scoring(True)
OUT = "results/analyses/benchA_benchB_exact"; os.makedirs(OUT, exist_ok=True)
for name, loader, g, tol in [("ovarian", load_ovarian, OV_CT1 + OV_CT2, 0.01), ("synthetic", load_sim, SIM_CT1 + SIM_CT2, 0.05)]:
    H = loader()
    cfg = RunConfig(encoding=EncodingConfig(norm="l1"), backend=BackendConfig(shots=5000),
                    search=SearchConfig(algorithm="multi_epoch", threshold=0.005, search_mode="pos", kl_tol=tol),
                    optimizer=OptimizerConfig(method="COBYLA", initial_angle=np.pi / 2))
    with contextlib.redirect_stdout(io.StringIO()):
        A = compare_scopes(cfg, H["ct1_mo"], H["ct2_mo"], H["ct1_co"], H["ct2_co"], gene_list=g)
    A["summary"].to_csv(f"{OUT}/{name}_scope_summary.csv", index=False)
    print("==", name); print(A["summary"].round(4).to_string())
    for sc, r in A["detail"].items():
        r["cumulative_ablation"].to_csv(f"{OUT}/{name}_{sc}_cumulative.csv", index=False)
        r["isolated_contribution"].to_csv(f"{OUT}/{name}_{sc}_isolated_loo.csv", index=False)
    with contextlib.redirect_stdout(io.StringIO()):
        Bm = benchmark_budget_matched(cfg, H["ct1_mo"], H["ct2_mo"], H["ct1_co"], H["ct2_co"], gene_list=g,
                                      tn_bond_dims=(1, 2), born_bond_dims=(1, 2),
                                      qx_best_effort=True, qx_kl_tol=0.01)
    Bm["sweep"].to_csv(f"{OUT}/{name}_benchB_sweep.csv", index=False)
    print("best-effort topology:", [f"{g[a]}->{g[b]}" for a, b in Bm["topology"]])
    print(Bm["sweep"].round(4).to_string())
