# liana_benchmark.py — LIANA+ DIFFERENTIAL cell-cell communication (Co vs Mo).
#
# LIANA's rank_aggregate is single-context (no built-in differential test). The
# LIANA+ route for "which LR interactions change between two conditions" is:
#   per-cell-type differential expression  ->  df_to_lr maps DE stats onto LR pairs
# (the LIANA+ analog of CellChat's netMappingDEG). We compute per-cell-type
# Co-vs-Mo DE with scanpy Wilcoxon, which works WITHOUT biological replicates
# (this dataset has one sample per condition, so pseudobulk+PyDESeq2 would have
# no replication — see the commented alternative for when replicates exist).
#
# Outputs to results/ccc_comparison/:
#   liana_{Mo,Co}_all.csv        per-condition rank_aggregate rankings (reference)
#   liana_co_vs_mo_diff.csv      LR interactions scored by Co-vs-Mo differential signal
#
# NOTE: df_to_lr's exact kwargs/column names vary by liana version — verify against
# your installed liana (>=1.4). The output schema below is standardized here so the
# collector does not depend on df_to_lr internals.
# ENV: conda activate ccc-liana   (liana>=1.4, decoupler, pydeseq2)
import os
import numpy as np
import pandas as pd
import scanpy as sc
import liana as li

try:                       # df_to_lr namespace moved across versions
    from liana.multi import df_to_lr
except Exception:
    from liana.mt import df_to_lr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT  = os.path.join(ROOT, "results", "ccc_comparison")
DATA = os.path.join(ROOT, "dataset_co_mo2.h5ad")
os.makedirs(OUT, exist_ok=True)

adata = sc.read_h5ad(DATA)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.obs["condition"] = adata.obs["BatchID"].str.extract(r"\((Co|Mo)\)")[0]
adata.obs["celltype"]  = adata.obs["BatchID"].str.replace(r"\s*\((Co|Mo)\)", "", regex=True)

# --- per-condition single-context ranking (kept for reference) ---------------
res_by = {}
for cond in ["Mo", "Co"]:
    sub = adata[adata.obs["condition"] == cond].copy()
    li.mt.rank_aggregate(sub, groupby="celltype", expr_prop=0.1, use_raw=False, verbose=True)
    sub.uns["liana_res"].to_csv(os.path.join(OUT, f"liana_{cond}_all.csv"), index=False)
    res_by[cond] = sub.uns["liana_res"]
    print(f"LIANA {cond}: {len(res_by[cond])} interactions")

# --- DIFFERENTIAL: per-cell-type Co-vs-Mo DE -> df_to_lr ---------------------
# Positive stat / log2FC = up in Co relative to Mo.
dea_rows = []
for ct in adata.obs["celltype"].unique():
    sub = adata[adata.obs["celltype"] == ct].copy()
    if sub.obs["condition"].nunique() < 2:
        continue
    sc.tl.rank_genes_groups(sub, groupby="condition", groups=["Co"], reference="Mo",
                            method="wilcoxon")
    de = sc.get.rank_genes_groups_df(sub, group="Co")   # names, scores, logfoldchanges, pvals, pvals_adj
    de = de.rename(columns={"names": "gene", "scores": "stat",
                            "logfoldchanges": "log2FoldChange",
                            "pvals": "pvalue", "pvals_adj": "padj"})
    de["celltype"] = ct
    dea_rows.append(de)
dea_df = pd.concat(dea_rows, ignore_index=True).set_index("gene")

lr = df_to_lr(
    adata,
    dea_df=dea_df,
    groupby="celltype",
    stat_keys=["stat", "pvalue", "padj"],
    resource_name="consensus",
    expr_prop=0.1,
    use_raw=False,
    complex_col="stat",         # aggregate complex subunits by this stat
    return_all_lrs=False,
    verbose=True,
)

# ---- standardize output so the collector is version-stable ------------------
# df_to_lr returns source/target/ligand_complex/receptor_complex + interaction
# stat columns (naming varies). Expose a single 'interaction_stat' (higher = more
# up in Co).
stat_col = next((c for c in ["interaction_stat", "lr_stat", "stat", "interaction_stat_mean"]
                 if c in lr.columns), None)
if stat_col is None:
    cand = [c for c in lr.columns if c.endswith("_stat")]
    lr["interaction_stat"] = lr[cand].mean(axis=1) if cand else np.nan
    stat_col = "interaction_stat"
if stat_col != "interaction_stat":
    lr = lr.rename(columns={stat_col: "interaction_stat"})

keep = [c for c in ["source", "target", "ligand_complex", "receptor_complex",
                    "interaction_stat", "interaction_pvalue", "interaction_padj"]
        if c in lr.columns]
lr_out = lr[keep].sort_values("interaction_stat", ascending=False)
out = os.path.join(OUT, "liana_co_vs_mo_diff.csv")
lr_out.to_csv(out, index=False)
print(f"LIANA+ (Co vs Mo df_to_lr): saved {len(lr_out)} LR interactions -> {out}")

# --- ALTERNATIVE (replicates available): pseudobulk + PyDESeq2 ---------------
# If you have >=2 replicate samples per condition, prefer this over Wilcoxon:
#   import decoupler as dc
#   from pydeseq2.dds import DeseqDataSet; from pydeseq2.ds import DeseqStats
#   pdata = dc.get_pseudobulk(adata, sample_col='sample', groups_col='celltype',
#                             layer='counts', mode='sum', min_cells=10, min_counts=1000)
#   # per celltype: DeseqDataSet(condition ~ Co/Mo) -> DeseqStats -> results_df
#   # concat into dea_df (index=gene, column 'celltype', columns stat/pvalue/padj)
#   # then the same df_to_lr(...) call above.
