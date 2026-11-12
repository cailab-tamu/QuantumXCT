"""
Review checks A-E (exact probabilities, shot noise, % of total reduction,
local-only null, random-topology null) on the topologies already saved in
results/.  Writes CSV/JSON to results/analyses/.
"""
import json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qxct import vector_normalize_dictionary_values as vn
from qxct.exact import ExactScorer, exact_contribution_table, shot_noise_samples
from qxct.scopes import enumerate_cnots
from analysis_tools.data import load_sim, load_ovarian, SIM_CT1, SIM_CT2, OV_CT1, OV_CT2

OUT = "results/analyses"; os.makedirs(OUT, exist_ok=True)
N_RANDOM = int(os.environ.get("N_RANDOM", 100))   # SI: 100 random topologies


def scorer_for(H, norm="l1"):
    return ExactScorer(vn(H["ct1_mo"], norm), vn(H["ct2_mo"], norm),
                       vn(H["ct1_co"], norm), vn(H["ct2_co"], norm))


def topo_from_json(p):
    d = json.load(open(p))
    return [tuple(x) for x in d["cnot_sequence"]], d["crx_angles"], d["gene_list"]


def random_null(sc, n_gates, pool, n=200, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        idx = rng.choice(len(pool), size=n_gates, replace=False)
        topo = [pool[i] for i in idx]
        _, k = sc.optimize_angles(topo)
        out.append(k)
    return np.array(out)


summary = {}
for name, loader, ct1, ct2, run_json in [
    ("synthetic", load_sim, SIM_CT1, SIM_CT2, "results/classic_sim_oo.json"),
    ("ovarian", load_ovarian, OV_CT1, OV_CT2, "results/ovarian_cocultured/run.json"),
]:
    t0 = time.time()
    H = loader()
    genes = ct1 + ct2
    sc = scorer_for(H, "l1")
    topo, ang, gl = topo_from_json(run_json)
    assert gl == genes, (gl, genes)

    # A/B: exact contribution table + shot noise at 5000 shots
    df, meta = exact_contribution_table(sc, topo, ang, genes, nshots=5000, n_rep=300)
    df.to_csv(f"{OUT}/{name}_exact_contributions.csv", index=False)
    base_samples = shot_noise_samples(sc.init1, sc.init2, sc.target1, sc.target2,
                                      [], [], shots=5000, n_rep=300, seed=1)
    meta["shot_sd_baseline_kl"] = float(np.std(base_samples))
    meta["shot_mean_baseline_kl"] = float(np.mean(base_samples))
    meta["n_gates"] = len(topo)

    # re-optimize angles on the exact objective (does the reported optimum hold?)
    ang_ex, kl_ex = sc.optimize_angles(topo, restarts=10)
    meta["final_kl_exact_reoptimized"] = kl_ex
    meta["angles_exact_reoptimized"] = ang_ex

    # C: local-only null (one RY per qubit, no two-qubit gates)
    _, kl_local = sc.local_ry_baseline(restarts=30)
    meta["local_RY_only_kl"] = kl_local
    meta["local_RY_only_params"] = sc.n

    # D: random-topology null with the same number of gates
    n1 = len(ct1)
    pool_all = enumerate_cnots(sc.n, n1, "all")
    pool_inter = enumerate_cnots(sc.n, n1, "inter")
    pool_intra = enumerate_cnots(sc.n, n1, "intra")
    for tag, pool in [("all", pool_all), ("inter", pool_inter), ("intra", pool_intra)]:
        k = min(len(topo), len(pool))
        null = random_null(sc, k, pool, n=N_RANDOM, seed=7)
        meta[f"random_{tag}_{k}gates_median"] = float(np.median(null))
        meta[f"random_{tag}_{k}gates_p05"] = float(np.percentile(null, 5))
        meta[f"random_{tag}_{k}gates_min"] = float(null.min())
        meta[f"random_{tag}_frac_better_or_equal_than_found"] = float(np.mean(null <= meta["final_kl"]))
        pd.Series(null).to_csv(f"{OUT}/{name}_random_null_{tag}.csv", index=False)

    # E: best intra-only (GRN) topology of the same size (exhaustive if small)
    meta["runtime_s"] = time.time() - t0
    summary[name] = meta
    print(name, json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in meta.items()
                            if not isinstance(v, list)}, indent=1))
    print(df.round(4).to_string())

json.dump(summary, open(f"{OUT}/exact_summary.json", "w"), indent=1, default=float)
