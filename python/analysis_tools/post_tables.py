"""Derived review tables that the SI quotes (previously computed interactively):
  - synthetic_recovery_corrected.csv   (edge recovery vs the corrected ground truth)
  - synthetic_single_edge_scores.csv   (exact single-gate score for every ordered pair)
  - ovarian_bootstrap_hub_frequency.csv / ovarian_bootstrap_target_frequency.csv
Run after run_search_exact.py (sim_basic, ovarian_boot)."""
import os, sys
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, networkx as nx
from qxct import vector_normalize_dictionary_values as vn
from qxct.exact import ExactScorer
from qxct.scopes import enumerate_cnots, scope_of_cnot
from analysis_tools.data import load_sim, SIM_CT1, SIM_CT2
from analysis_tools.recovery import evaluate, GT_DIRECT
R = "results/analyses"
genes = SIM_CT1 + SIM_CT2; ix = {g: i for i, g in enumerate(genes)}; ids = [int(g.split("_")[1]) for g in genes]

# 1) recovery of the exact-search topologies (+ the saved sampled run)
rows = []
d = pd.read_csv(f"{R}/synthetic_exact_scopes_norms.csv")
for _, r in d.iterrows():
    edges = [tuple(e.split("->")) for e in str(r.edges).split(";") if "->" in e]
    o = evaluate([(ix[a], ix[b]) for a, b in edges], genes, f"{r.norm}-{r.scope}")
    o.update(norm=r.norm, scope=r.scope, final=r.final, pct=r.pct); rows.append(o)
import json
old = json.load(open("results/classic_sim_oo.json"))
rows.append(evaluate([tuple(x) for x in old["cnot_sequence"]], genes, "saved l1-all (sampled)"))
pd.DataFrame(rows).to_csv(f"{R}/synthetic_recovery_corrected.csv", index=False)

# 2) single-edge scores over all ordered pairs
G = nx.DiGraph(GT_DIRECT); rows = []
H = load_sim()
for norm in ["l1", "l2"]:
    sc = ExactScorer(*[vn(H[k], norm) for k in ("ct1_mo", "ct2_mo", "ct1_co", "ct2_co")]); b = sc.baseline()
    for c, t in enumerate_cnots(6, 2, "all"):
        _, k = sc.optimize_angles([(c, t)], restarts=3)
        a, z = ids[c], ids[t]
        rows.append(dict(norm=norm, edge=f"{genes[c]}->{genes[t]}", scope=scope_of_cnot(c, t, 2), score=b - k,
                         direct=int((a, z) in GT_DIRECT),
                         closure=int(a in G and z in G and nx.has_path(G, a, z))))
pd.DataFrame(rows).to_csv(f"{R}/synthetic_single_edge_scores.csv", index=False)

# 3) bootstrap hub / target frequencies (ovarian)
b = pd.read_csv(f"{R}/ovarian_boot.csv"); b.edges = b.edges.fillna("")
ct1 = {"PDGFB", "STAT3", "HES4", "TGFB1"}
E = lambda s: [tuple(x.split("->")) for x in s.split(";") if x]
tests = {
    "PDGFB<->PDGFRB (either dir)": lambda es: any({a, c} == {"PDGFB", "PDGFRB"} for a, c in es),
    "PDGFRB->PDGFB": lambda es: ("PDGFRB", "PDGFB") in es,
    "any cancer->PDGFRB": lambda es: any(c == "PDGFRB" and a in ct1 for a, c in es),
    "hub {PDGFB,PDGFRB,STAT3} >=2 edges": lambda es: sum(1 for a, c in es if {a, c} <= {"PDGFB", "PDGFRB", "STAT3"}) >= 2,
    "any TGF-beta gene edge": lambda es: any(a.startswith("TGF") or c.startswith("TGF") for a, c in es),
}
hub = {n: {k: round(np.mean([f(E(x)) for x in s.edges]), 2) for k, f in tests.items()} for n, s in b.groupby("norm")}
pd.DataFrame(hub).to_csv(f"{R}/ovarian_bootstrap_hub_frequency.csv")
tg = {}
for n, s in b.groupby("norm"):
    tg[n] = {g: np.mean([any(c == g and ((a in ct1) != (g in ct1)) for a, c in E(x)) for x in s.edges])
             for g in ["PDGFB", "STAT3", "HES4", "TGFB1", "PDGFRB", "TGFBR1", "TGFBR2"]}
pd.DataFrame(tg).to_csv(f"{R}/ovarian_bootstrap_target_frequency.csv")
print("post tables written")
