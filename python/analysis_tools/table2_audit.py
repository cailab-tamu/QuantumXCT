"""Recompute manuscript Table 2 (L2 encoding) with exact probabilities and the
correct denominator (total KL reduction)."""
import os, sys; sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
from qxct import vector_normalize_dictionary_values as vn
from qxct.exact import ExactScorer, exact_contribution_table
from analysis_tools.data import load_ovarian, OV_CT1, OV_CT2
g=OV_CT1+OV_CT2; ix={x:i for i,x in enumerate(g)}
H=load_ovarian(); sc=ExactScorer(*[vn(H[k],"l2") for k in ("ct1_mo","ct2_mo","ct1_co","ct2_co")])
T={"N-wise":[("PDGFB","HES4",3.1416),("PDGFB","PDGFRB",1.0199),("PDGFRB","PDGFB",2.5114),("PDGFB","TGFBR2",1.9131),("HES4","PDGFRB",1.5732),("TGFBR2","PDGFB",0.0),("STAT3","HES4",1.2655),("PDGFRB","STAT3",1.7114)],  # 8-gate exact L2 all (current Fig 3A)
   "Multi-epoch":[("STAT3","PDGFRB",1.416),("PDGFRB","PDGFB",1.433),("PDGFB","PDGFRB",2.260),("PDGFRB","STAT3",1.3e-7),("PDGFB","TGFBR2",1.8e-5)]}
out=[]
for alg,rows in T.items():
    topo=[(ix[a],ix[b]) for a,b,_ in rows]; ang=[t for *_,t in rows]
    for tag,angles in [("manuscript angles",ang),("exact re-optimized",sc.optimize_angles(topo,restarts=10)[0])]:
        df,meta=exact_contribution_table(sc,topo,angles,g,nshots=5000,n_rep=300)
        df.insert(0,"angles",tag); df.insert(0,"algorithm",alg)
        for k in ("baseline_kl","final_kl","total_reduction","pct_reduction","shot_sd_final_kl"): df[k]=meta[k]
        out.append(df); print(alg,tag,{k:round(v,4) for k,v in meta.items()}); print(df[["gate","theta_rad","KL_cumulative","dKL_step","pct_of_total_reduction_step","isolated_reduction","LOO_increase","pct_LOO_of_total_reduction"]].round(4).to_string())
pd.concat(out).to_csv("results/analyses/table2_exact_audit_L2.csv",index=False)
