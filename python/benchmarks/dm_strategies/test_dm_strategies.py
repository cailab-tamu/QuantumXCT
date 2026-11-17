"""Validate candidate-equivalence across DM-difference strategies on small N."""
# --- Run from the QuantumXCT/python root wherever this file is launched from, so
#     inputs (dataset_*.h5ad) are found there and outputs go to python/results/.
import os as _os, sys as _sys
_PY_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
_os.chdir(_PY_ROOT)
if _PY_ROOT not in _sys.path:
    _sys.path.insert(0, _PY_ROOT)
# ---
import os, sys, time, tracemalloc
import numpy as np
from qxct.circuits import (find_cnot_candidates_from_state_diff,
                           _process_target_state_input, create_cnot_pairs_from_locations)
from qxct.encoding import create_joint_histogram

def synth(n, seed, shift):
    rng=np.random.default_rng(seed); p=np.clip(rng.uniform(.3,.7,n)+shift,.05,.95)
    X=(rng.random((300,n))<p).astype(int); h,_=create_joint_histogram(X); return h

# small sim-like data: 2 + 4 qubits
i1=synth(2,1,0.0); i2=synth(4,2,0.0); t1=synth(2,1,0.25); t2=synth(4,2,0.25)
THRESH=0.005; MODE="pos"

def _pairs_from_indices(idx_pairs, num_qubits):
    labels=[bin(i)[2:].zfill(num_qubits) for i in range(2**num_qubits)]
    cfg=set()
    for r,c in idx_pairs:
        lr,lc=labels[r],labels[c]
        qr=[num_qubits-1-i for i,b in enumerate(lr) if b=='1']
        qc=[num_qubits-1-i for i,b in enumerate(lc) if b=='1']
        if qr or qc: cfg.update(create_cnot_pairs_from_locations(qr,qc))
    return sorted(cfg)

def amps(dtype):
    a1,_=_process_target_state_input(i1,None); a2,_=_process_target_state_input(i2,None)
    b1,_=_process_target_state_input(t1,None); b2,_=_process_target_state_input(t2,None)
    a0=np.kron(a2,a1).real.astype(dtype); a=np.kron(b2,b1).real.astype(dtype)
    return a0,a

def peaks_mask(D):
    if MODE=="pos": r,c=np.where(D>THRESH)
    elif MODE=="neg": r,c=np.where(D<-THRESH)
    else: r,c=np.where(np.abs(D)>THRESH)
    return list(zip(r.tolist(),c.tolist()))

def strat_dense(dtype):
    a0,a=amps(dtype)
    N=int(np.log2(len(a)))
    D=(np.outer(a,a)-np.outer(a0,a0))
    return _pairs_from_indices(peaks_mask(D), N)

def strat_list():
    # scalable: sparse dicts, iterate union support, keep passing pairs directly (no matrix)
    a1,_=_process_target_state_input(i1,None); a2,_=_process_target_state_input(i2,None)
    b1,_=_process_target_state_input(t1,None); b2,_=_process_target_state_input(t2,None)
    a0=np.kron(a2,a1).real; a=np.kron(b2,b1).real
    N=int(np.log2(len(a)))
    S=sorted(set(np.nonzero(a0)[0]).union(np.nonzero(a)[0]))
    idx=[]
    for r in S:
        for c in S:
            d=a[r]*a[c]-a0[r]*a0[c]
            if (MODE=="pos" and d>THRESH) or (MODE=="neg" and d<-THRESH) or (MODE=="abs" and abs(d)>THRESH):
                idx.append((r,c))
    return _pairs_from_indices(idx,N), len(S)

def timed(fn):
    tracemalloc.start(); t=time.perf_counter(); out=fn(); dt=time.perf_counter()-t
    peak=tracemalloc.get_traced_memory()[1]; tracemalloc.stop(); return out,dt,peak

# A: current implementation (complex128 dense)
cA,dtA,mA=timed(lambda: find_cnot_candidates_from_state_diff(i1,i2,t1,t2,threshold=THRESH,search_mode=MODE,norm=None)[0])
# B: dense float32
cB,dtB,mB=timed(lambda: strat_dense(np.float32))
# C: dense float64 (real)
cC,dtC,mC=timed(lambda: strat_dense(np.float64))
# D: direct list from sparse support
(dD,SS),dtD,mD=timed(strat_list); cD=dD

print("N qubits:",6," | threshold:",THRESH," | mode:",MODE," | |S| support:",SS)
print(f"{'strategy':<26}{'#cands':>7}{'time_ms':>10}{'peakKB':>10}  match_current")
for name,c,dt,m in [("A current (complex128)",cA,dtA,mA),("B dense float32",cB,dtB,mB),
                    ("C dense float64",cC,dtC,mC),("D direct list (sparse)",cD,dtD,mD)]:
    print(f"{name:<26}{len(c):>7}{dt*1000:>10.2f}{m/1024:>10.1f}  {c==cA}")
print("\nALL identical to current:", cB==cA and cC==cA and cD==cA)
