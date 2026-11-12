"""Run the package's own searches with exact scoring: ovarian/synthetic L1 vs L2,
GRN / LR / GRN+LR scopes, both search algorithms (multi_epoch and iterative_n_wise),
and cell-bootstrap edge stability (multi_epoch).

Outputs (results/analyses):
  {ovarian,synthetic}_exact_scopes_norms.csv       multi_epoch only (main Tables 2/4, SI S1)
  {ovarian,synthetic}_exact_scopes_algorithms.csv  both algorithms, with an 'algorithm' column
"""
import os, sys, json, time, io, contextlib
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
from qxct import QuantumXct, RunConfig, EncodingConfig, BackendConfig, SearchConfig, OptimizerConfig
from qxct.exact import use_exact_scoring, ExactScorer
from qxct import vector_normalize_dictionary_values as vn
from qxct.encoding import create_joint_histogram
from analysis_tools.data import load_ovarian, load_sim, OV_CT1, OV_CT2, SIM_CT1, SIM_CT2
use_exact_scoring(True)
OUT="results/analyses"; os.makedirs(OUT, exist_ok=True)
ALGOS = ["multi_epoch", "iterative_n_wise"]
def cfg(norm="l1", seed=42, kl_tol=0.01, scope="all", algorithm="multi_epoch"):
    return RunConfig(encoding=EncodingConfig(norm=norm), backend=BackendConfig(shots=5000),
        search=SearchConfig(algorithm=algorithm, threshold=0.005, search_mode="pos",
                            n_epochs=None, kl_tol=kl_tol, max_greedy_depth=30, candidate_scope=scope),
        optimizer=OptimizerConfig(method="COBYLA", initial_angle=np.pi/2), seed=seed)
def run(H, genes, c):
    qx=QuantumXct(c, gene_list=genes)
    qx.encode(H["ct1_mo"],H["ct2_mo"],H["ct1_co"],H["ct2_co"])
    with contextlib.redirect_stdout(io.StringIO()):
        qx.screen_candidates()
        if c.search.candidate_scope=="intra" and c.search.algorithm=="multi_epoch": qx.search_topology_grn_sequential()
        else: qx.search_topology()
        qx.optimize_angles()
    sc=ExactScorer(qx.initial1,qx.initial2,qx.target1,qx.target2)
    return qx.result.topology, qx.result.angles, sc.kl(qx.result.topology,qx.result.angles), sc.baseline()
def lab(genes,t): return [f"{genes[a]}->{genes[b]}" for a,b in t]
mode=sys.argv[1] if (__name__=="__main__" and len(sys.argv)>1) else None
if mode=="ovarian_basic":
    H=load_ovarian(); g=OV_CT1+OV_CT2; rows=[]
    for algo in ALGOS:
        for norm in ["l1","l2"]:
            for scope in ["all","intra","inter"]:
                t0=time.time(); topo,ang,k,b=run(H,g,cfg(norm,scope=scope,algorithm=algo))
                rows.append(dict(algorithm=algo,norm=norm,scope=scope,edges=";".join(lab(g,topo)),n=len(topo),base=b,final=k,pct=100*(b-k)/b,sec=time.time()-t0)); print(rows[-1],flush=True)
    df=pd.DataFrame(rows)
    df.to_csv(f"{OUT}/ovarian_exact_scopes_algorithms.csv",index=False)
    df[df.algorithm=="multi_epoch"].drop(columns="algorithm").to_csv(f"{OUT}/ovarian_exact_scopes_norms.csv",index=False)
if mode=="sim_basic":
    H=load_sim(); g=SIM_CT1+SIM_CT2; rows=[]
    for algo in ALGOS:
        for norm in ["l1","l2"]:
            for scope in ["all","intra","inter"]:
                t0=time.time(); topo,ang,k,b=run(H,g,cfg(norm,kl_tol=0.05,scope=scope,algorithm=algo))
                rows.append(dict(algorithm=algo,norm=norm,scope=scope,edges=";".join(lab(g,topo)),n=len(topo),base=b,final=k,pct=100*(b-k)/b,sec=time.time()-t0)); print(rows[-1],flush=True)
    df=pd.DataFrame(rows)
    df.to_csv(f"{OUT}/synthetic_exact_scopes_algorithms.csv",index=False)
    df[df.algorithm=="multi_epoch"].drop(columns="algorithm").to_csv(f"{OUT}/synthetic_exact_scopes_norms.csv",index=False)
if mode in ("ovarian_boot","sim_boot"):
    B=int(os.environ.get("NBOOT", 50 if mode=="ovarian_boot" else 20))   # SI: 50 ovarian / 20 synthetic
    if mode=="ovarian_boot": (H,X)=load_ovarian(return_matrices=True); g=OV_CT1+OV_CT2; tol=0.01
    else: (H,X)=load_sim(return_matrices=True); g=SIM_CT1+SIM_CT2; tol=0.05
    rng=np.random.default_rng(2026); rows=[]
    for b in range(B):
        Hb={k:dict(create_joint_histogram((v[rng.integers(0,len(v),len(v))]>0).astype(int))[0]) for k,v in X.items()}
        for norm in ["l1","l2"]:
            topo,ang,k,base=run(Hb,g,cfg(norm,seed=1000+b,kl_tol=tol))
            rows.append(dict(boot=b,norm=norm,edges=";".join(lab(g,topo)),n=len(topo),base=base,final=k))
        print(b,rows[-2]["edges"],"|",rows[-1]["edges"],flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/{mode}.csv",index=False)
