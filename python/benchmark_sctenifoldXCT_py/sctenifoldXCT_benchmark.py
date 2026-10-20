# sctenifoldXCT_benchmark.py — DIFFERENTIAL cell-cell communication.
# scTenifoldXct is a *differential* method: build one object per condition and
# compare them. This mirrors the QuantumXCT mono-vs-co design — it reports the
# ligand-receptor pairs whose Cancer Cells -> Fibroblasts interaction CHANGES
# between co-culture (Co) and mono-culture (Mo).
# ENV: run inside the dedicated scTenifoldXct env (see CCC_SETUP_README.md), NOT qiskit-env.
import os
import scanpy as sc
import scTenifoldXct as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                                  # .../QuantumXCT/python
OUT  = os.path.join(ROOT, "results", "ccc_comparison")
DATA = os.path.join(ROOT, "dataset_co_mo2.h5ad")
GRN  = os.path.join(HERE, "grn_cache")                        # per-condition GRN cache
os.makedirs(OUT, exist_ok=True)

SOURCE, TARGET = "Cancer Cells", "Fibroblasts"               # ligand cell -> receptor cell

adata = sc.read_h5ad(DATA)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Top 3000 HVGs, computed ONCE on the full object so both conditions share the
# same feature space (scTenifoldXct's merge requires aligned genes) and so the
# per-cell-type GRN construction stays tractable.
sc.pp.highly_variable_genes(adata, n_top_genes=3000)
adata = adata[:, adata.var.highly_variable].copy()

# celltype = source/target labels; condition = Co vs Mo (both from BatchID)
adata.obs["celltype"]  = adata.obs["BatchID"].str.replace(r"\s*\((Co|Mo)\)", "", regex=True)
adata.obs["condition"] = adata.obs["BatchID"].str.extract(r"\((Co|Mo)\)")[0]

ada_co = adata[adata.obs["condition"] == "Co"].copy()
ada_mo = adata[adata.obs["condition"] == "Mo"].copy()

# One scTenifoldXct object per condition (same source/target cell types). Each
# builds its own per-cell-type GRN, cached to a separate dir; set rebuild_GRN=False
# on a re-run to load the cached GRNs in seconds instead of rebuilding.
xct_co = st.scTenifoldXct(data=ada_co, source_celltype=SOURCE, target_celltype=TARGET,
                          obs_label="celltype", rebuild_GRN=True,
                          GRN_file_dir=os.path.join(GRN, "co"), verbose=True, n_cpus=-1)
xct_mo = st.scTenifoldXct(data=ada_mo, source_celltype=SOURCE, target_celltype=TARGET,
                          obs_label="celltype", rebuild_GRN=True,
                          GRN_file_dir=os.path.join(GRN, "mo"), verbose=True, n_cpus=-1)

# Merge the two conditions (Co vs Mo) and train the manifold-alignment embedding.
XCTs = st.merge_scTenifoldXct(xct_co, xct_mo)
emb = XCTs.get_embeds(train=True)

# Differential test: LR pairs whose interaction differs Co vs Mo.
XCTs.nn_aligned_diff(emb)
pairs_diff = XCTs.chi2_diff_test()

out_csv = os.path.join(OUT, "sctenifoldxct_co_vs_mo_diff.csv")
pairs_diff.to_csv(out_csv)
print(f"scTenifoldXct (Co vs Mo differential): saved {len(pairs_diff)} pairs -> {out_csv}")
