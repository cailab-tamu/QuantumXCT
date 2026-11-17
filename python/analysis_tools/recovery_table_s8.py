"""Reproduce SI Table S8 (synthetic edge/direction recovery) from the saved
exact topologies in results/analyses/synthetic_exact_scopes_algorithms.csv.

No search is run here: recovery is purely the predicted edge set vs the
simulator ground truth (edge counting), so it needs no qiskit/scanpy.

Run from QuantumXCT/python:
    python analysis_tools/recovery_table_s8.py
"""
import os, sys, math
sys.path.insert(0, os.getcwd())
import pandas as pd

# --- simulator ground truth (simCellsSpace/simulate_cells.ipynb) ---
GT = [(50, 60), (60, 70), (70, 71), (70, 72), (70, 80), (80, 90),
      (73, 74), (73, 75), (73, 76)]
# the 6-gene synthetic panel  {g50, g90 | g60, g70, g71, g80}
PANEL = {50, 60, 70, 71, 80, 90}

# directed reachability in the ground-truth graph (for the causal-path closure)
_adj = {}
for a, b in GT:
    _adj.setdefault(a, []).append(b)

def _reaches(a, b):
    seen, st = {a}, [a]
    while st:
        x = st.pop()
        for y in _adj.get(x, []):
            if y == b:
                return True
            if y not in seen:
                seen.add(y); st.append(y)
    return False

DIRECT = {e for e in GT if e[0] in PANEL and e[1] in PANEL}                 # |G| = 5
CLOSURE = {(a, b) for a in PANEL for b in PANEL if a != b and _reaches(a, b)}  # |G| = 13
N = len(PANEL) * (len(PANEL) - 1)                                          # 30 ordered pairs

def _hyp(tp, K, n):
    """Pr[X >= tp], X ~ Hypergeometric(population=N, successes=K, draws=n)."""
    return sum(math.comb(K, i) * math.comb(N - K, n - i) / math.comb(N, n)
               for i in range(tp, min(K, n) + 1))

def recovery(edges):
    pred = list(dict.fromkeys(edges))          # dedup, keep order
    n = len(pred)
    out = {}
    for name, S in [("direct", DIRECT), ("closure", CLOSURE)]:
        tp = len([e for e in pred if e in S])                              # true positives
        K = len(S)
        und = len([e for e in pred if e in S or e[::-1] in S])             # hits ignoring direction
        out[name] = dict(tp=tp, E=n, recall=tp / K,
                         dir_acc=(tp / und if und else float("nan")),
                         p=_hyp(tp, K, n))
    return out

def _edges(cell):
    return [tuple(int(x.split("_")[1]) for x in e.split("->")) for e in cell.split(";")]

if __name__ == "__main__":
    d = pd.read_csv("results/analyses/synthetic_exact_scopes_algorithms.csv")
    # the four exact multi-epoch topologies reported in Table S8
    ROWS = [("l1", "all"), ("l1", "inter"), ("l2", "all"), ("l2", "inter")]
    hdr = f"{'Topology':<38}{'TP/|E|':>8}{'recall':>8}{'p_dir':>7}   {'TP/|E|':>8}{'dir.acc':>8}{'p_clo':>7}"
    print(f"|G|: direct = {len(DIRECT)}, closure = {len(CLOSURE)}, N(N-1) = {N}\n")
    print(hdr); print("-" * len(hdr))
    for norm, scope in ROWS:
        r = d[(d.algorithm == "multi_epoch") & (d.norm == norm) & (d.scope == scope)].iloc[0]
        rec = recovery(_edges(r["edges"]))
        D, C = rec["direct"], rec["closure"]
        lab = f"Exact {norm.upper()}, {scope} scope ({int(r['n'])} gates)"
        print(f"{lab:<38}{D['tp']}/{D['E']:<6}{D['recall']:>8.2f}{D['p']:>7.2f}   "
              f"{C['tp']}/{C['E']:<6}{C['dir_acc']:>8.2f}{C['p']:>7.2f}")
