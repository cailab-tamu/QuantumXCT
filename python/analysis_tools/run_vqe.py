import os, sys, io, contextlib, time, json
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
from qxct import QuantumXct, RunConfig, EncodingConfig, BackendConfig, SearchConfig, OptimizerConfig
from qxct.exact import use_exact_scoring, ExactScorer
from analysis_tools.data import load_ovarian, OV_CT1, OV_CT2
use_exact_scoring(True)
H=load_ovarian(); g=OV_CT1+OV_CT2; rows=[]
for algo in ["multi_epoch","iterative_n_wise","vqe_hybrid"]:
    c=RunConfig(encoding=EncodingConfig(norm="l1"), backend=BackendConfig(shots=5000),
        search=SearchConfig(algorithm=algo, threshold=0.02, search_mode="pos", kl_tol=0.01, vqe_niter=150),
        optimizer=OptimizerConfig(method="COBYLA"))
    qx=QuantumXct(c, gene_list=g); qx.encode(H["ct1_mo"],H["ct2_mo"],H["ct1_co"],H["ct2_co"])
    t0=time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        cands=qx.screen_candidates(); qx.search_topology(); qx.optimize_angles()
    sc=ExactScorer(qx.initial1,qx.initial2,qx.target1,qx.target2); k=sc.kl(qx.result.topology,qx.result.angles)
    rows.append(dict(algorithm=algo,threshold=0.02,n_candidates=len(cands),n_gates=len(qx.result.topology),
        edges=";".join(f"{g[a]}->{g[b]}" for a,b in qx.result.topology),baseline=sc.baseline(),final=k,
        pct=100*(1-k/sc.baseline()),sec=time.time()-t0)); print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv("results/analyses/ovarian_algorithms_thr002_exact.csv",index=False)
