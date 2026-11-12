# data_prep/export_metadata.py
# Export the per-cell annotations (obs) and the gene list from an annotated .h5ad.
# The downloader (pull_ovarian.py) re-attaches these authors' labels (Cancer/Fibroblast,
# mono/co) to the raw counts pulled from GEO, so reviewers rebuild the exact dataset.
# Run once against your current h5ad:
#     python data_prep/export_metadata.py [path/to/dataset.h5ad]
import sys, os
import anndata as ad

H = sys.argv[1] if len(sys.argv) > 1 else "dataset_co_mo2.h5ad"
stem = os.path.splitext(os.path.basename(H))[0]
OUT = os.path.dirname(os.path.abspath(__file__))
A = ad.read_h5ad(H, backed="r")
obs = A.obs.copy()
obs.index.name = "barcode"
meta_csv = os.path.join(OUT, f"{stem}_cell_metadata.csv")
genes_txt = os.path.join(OUT, f"{stem}_genes.txt")
obs.to_csv(meta_csv)
with open(genes_txt, "w") as f:
    f.write("\n".join(map(str, A.var_names)))
print(f"wrote {meta_csv}  ({len(obs)} cells; columns {list(obs.columns)})")
print(f"wrote {genes_txt}  ({A.n_vars} genes)")
