"""Export dataset_co_mo2.h5ad to plain files a base-Seurat script can read.

Avoids the fragile sceasy / SeuratDisk h5ad readers: run this in the Python env
that already reads the .h5ad (e.g. qiskit-env with scanpy), then run
convert_h5ad_to_seurat.R (base Seurat only) to build dataset_co_mo2.rds.

Writes into this folder (benchmark_cellchat_r/):
    counts.mtx.gz      genes x cells raw counts (MatrixMarket, gzipped)
    features.tsv.gz    gene names (rows of the matrix)
    barcodes.tsv.gz    cell ids   (cols of the matrix)
    meta.csv           barcode, CellType, BatchID

Run from anywhere:
    python benchmark_cellchat_r/export_h5ad_for_cellchat.py
    # or point at a different file:
    python benchmark_cellchat_r/export_h5ad_for_cellchat.py /path/to/dataset_co_mo2.h5ad
"""
import os, sys, gzip
import numpy as np
import scanpy as sc
import scipy.io, scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))          # .../python/benchmark_cellchat_r
PYROOT = os.path.dirname(HERE)                             # .../python
h5ad = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PYROOT, "dataset_co_mo2.h5ad")

ad = sc.read_h5ad(h5ad)
X = ad.X
X = sp.csr_matrix(X) if not sp.issparse(X) else X
# Seurat ReadMtx expects genes x cells
M = sp.csc_matrix(X.T)
# raw counts -> integer field if they are whole numbers, else real
is_int = np.allclose(M.data, np.round(M.data))
field = "integer" if is_int else "real"
if is_int:
    M = M.astype(np.int64)

with gzip.open(os.path.join(HERE, "counts.mtx.gz"), "wb") as f:
    scipy.io.mmwrite(f, M, field=field)
with gzip.open(os.path.join(HERE, "features.tsv.gz"), "wt") as f:
    f.write("\n".join(map(str, ad.var_names)) + "\n")
with gzip.open(os.path.join(HERE, "barcodes.tsv.gz"), "wt") as f:
    f.write("\n".join(map(str, ad.obs_names)) + "\n")

for col in ("CellType", "BatchID"):
    if col not in ad.obs.columns:
        raise SystemExit(f"h5ad obs is missing required column '{col}'. Found: {list(ad.obs.columns)}")
ad.obs[["CellType", "BatchID"]].to_csv(os.path.join(HERE, "meta.csv"), index_label="barcode")

print(f"Exported {M.shape[0]} genes x {M.shape[1]} cells ({field} counts) to {HERE}")
print("Now run:  Rscript benchmark_cellchat_r/convert_h5ad_to_seurat.R")
