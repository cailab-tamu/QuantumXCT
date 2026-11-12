"""Pass-1 verification: every key number quoted in the revised main text / SI is recomputed from results files."""
import json, pandas as pd, re, sys
R="results/analyses"; import os
_cands=[os.environ.get("QXCT_PAPER_DIR",""), "../../../QuantumXCT_KL_paper/revision", "../paper"]
PAPER=next(d for d in _cands if d and os.path.exists(os.path.join(d,"sn-article_revised.tex")))
M=open(os.path.join(PAPER,"sn-article_revised.tex")).read(); S=open(os.path.join(PAPER,"sn-supplementary.tex")).read()
ok=0; bad=[]
def chk(val, fmt, where, label):
    global ok
    txt=fmt.format(val); doc=M if where=="M" else S
    if txt in doc: ok+=1
    else: bad.append((label, txt, where))
e=json.load(open(f"{R}/exact_summary.json"))
o,sy=e["ovarian"],e["synthetic"]
chk(o["baseline_kl"],"{:.3f}","S","ov base"); chk(o["final_kl"],"{:.4f}","S","ov final"); chk(o["final_kl_exact_reoptimized"],"{:.4f}","S","ov reopt")
chk(o["local_RY_only_kl"],"{:.4f}","S","ov local"); chk(sy["local_RY_only_kl"],"{:.4f}","S","sy local")
chk(sy["baseline_kl"],"{:.3f}","S","sy base"); chk(sy["final_kl"],"{:.3f}","S","sy final"); chk(sy["final_kl_exact_reoptimized"],"{:.3f}","S","sy reopt")
chk(o["random_all_4gates_median"],"{:.3f}","S","ov rnd med"); chk(o["random_inter_4gates_min"],"{:.3f}","S","ov rnd inter min")
chk(sy["random_inter_13gates_median"],"{:.3f}","S","sy rnd inter med"); chk(sy["random_intra_13gates_median"],"{:.3f}","S","sy rnd intra med")
chk(o["pct_reduction"],"{:.1f}","S","ov pct"); chk(o["shot_sd_final_kl"],"{:.4f}","S","sd final"); chk(o["shot_sd_baseline_kl"],"{:.4f}","S","sd base")
# (random / local-RY rows now live in SI Table S2 and are checked above)
c=pd.read_csv(f"{R}/ovarian_exact_contributions.csv")
for _,r in c.iterrows():
    chk(r.pct_of_total_reduction_step,"{:.1f}","S","S3 seq "+r.gate); chk(r.pct_LOO_of_total_reduction,"{:.1f}","S","S3 loo "+r.gate); chk(r.KL_cumulative,"{:.4f}","S","S3 kl "+r.gate)
t=pd.read_csv(f"{R}/table2_exact_audit_L2.csv"); t=t[t.angles=="exact re-optimized"]
for _,r in t.iterrows():   # main Table 3: theta, KL, % of initial KL, % of total reduction, isolated % of initial KL
    a_,b_=r.gate.split(" -> ")
    line=[l for l in M.splitlines() if f"& {a_} & {b_}" in l.replace("  "," ") and f"{r.theta_rad:.3f}" in l]
    vals=[f"{r.KL_cumulative:.3f}", -r.dKL_step/r.baseline_kl*100, r.pct_of_total_reduction_step, r.isolated_reduction/r.baseline_kl*100]
    for v in vals:
        txt=v if isinstance(v,str) else (f"$-{abs(v):.1f}$" if v<0 else f"{v:.1f}")
        if line and txt in line[0]: ok+=1
        else: bad.append(("T3 "+r.algorithm+" "+r.gate, txt, "M"))
sn=pd.read_csv(f"{R}/ovarian_exact_scopes_norms.csv")
for _,r in sn.iterrows(): chk(r.pct,"{:.1f}","S",f"S1 {r.norm} {r.scope}")
for _,r in sn[sn.norm=="l1"].iterrows(): chk(r.pct,"{:.1f}","M",f"T2 ov {r.scope}")
ss=pd.read_csv(f"{R}/synthetic_exact_scopes_norms.csv")
for _,r in ss[ss.norm=="l1"].iterrows(): chk(r.pct,"{:.1f}","M",f"T2 sy {r.scope}")
se=pd.read_csv(f"{R}/synthetic_single_edge_scores.csv")
from sklearn.metrics import roc_auc_score
for n in ["l1","l2"]:
    for sc_ in ["inter","intra"]:
        s=se[(se.norm==n)&(se.scope==sc_)]
        for lab in ["direct","closure"]: chk(roc_auc_score(s[lab],s.score),"{:.2f}","S",f"AUC {n} {sc_} {lab}")
hub=pd.read_csv(f"{R}/ovarian_bootstrap_hub_frequency.csv",index_col=0)
for key in ["any cancer->PDGFRB","PDGFRB->PDGFB"]:
    for n in ["l1","l2"]: chk(hub.loc[key,n],"{:.2f}","S",f"boot {key} {n}")
hw=json.load(open("results/hardware/hardware_postopt_report.json")); k=[x for x in hw if x.startswith("fake")][0]
chk(hw[k]["reduction_pct_readout_corrected"],"{:.1f}","S","hw red corr"); chk(hw[k]["reduction_pct_raw"],"{:.1f}","S","hw raw")
chk(hw[k]["optimized"]["transpiled_2q_gates"],"{}","S","hw 2q"); chk(hw[k]["optimized"]["transpiled_depth"],"{}","S","hw depth")
alg=pd.read_csv(f"{R}/ovarian_algorithms_thr002_exact.csv")
for _,r in alg.iterrows(): chk(r.pct,"{:.1f}","M","alg "+r.algorithm)
print("OK:",ok); print("MISMATCH:",len(bad))
for b in bad: print("  ",b)
sys.exit(1 if bad else 0)
