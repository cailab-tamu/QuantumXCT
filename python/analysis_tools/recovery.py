"""Edge recovery against the TRUE simulator network, with testable-edge and
transitive-closure definitions and a hypergeometric null."""
import json, itertools, sys, os
sys.path.insert(0, os.getcwd())
import networkx as nx
from scipy.stats import hypergeom
GT_DIRECT = [(50,60),(60,70),(70,71),(70,72),(70,80),(80,90),(73,74),(73,75),(73,76)]  # simCellsSpace/simulate_cells.ipynb
def evaluate(topology, genes, label):
    ids=[int(g.split('_')[1]) for g in genes]
    G=nx.DiGraph(GT_DIRECT)
    direct={e for e in GT_DIRECT if e[0] in ids and e[1] in ids}
    closure={(a,b) for a in ids for b in ids if a!=b and a in G and b in G and nx.has_path(G,a,b)}
    pred=list(dict.fromkeys((ids[c],ids[t]) for c,t in topology))
    N=len(ids)*(len(ids)-1); n=len(pred); out={"label":label,"n_pred":n,"n_possible":N}
    for nm,S in [("direct",direct),("closure",closure)]:
        tp=len([e for e in pred if e in S]); K=len(S)
        und=len([e for e in pred if e in S or e[::-1] in S])
        out.update({f"{nm}_K":K,f"{nm}_TP":tp,f"{nm}_precision":tp/n if n else float('nan'),
                    f"{nm}_recall":tp/K if K else float('nan'),
                    f"{nm}_undirected_hits":und,
                    f"{nm}_direction_acc":tp/und if und else float('nan'),
                    f"{nm}_null_expected_TP":n*K/N,
                    f"{nm}_hypergeom_p":float(hypergeom.sf(tp-1,N,K,n))})
    return out
if __name__=="__main__":
    d=json.load(open("results/classic_sim_oo.json"))
    r=evaluate([tuple(x) for x in d["cnot_sequence"]], d["gene_list"], "synthetic all-scope (saved run)")
    print(json.dumps(r,indent=1))
