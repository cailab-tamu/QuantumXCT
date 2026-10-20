# convert_h5ad_to_seurat.R  —  dataset_co_mo2.h5ad -> dataset_co_mo2.rds
#
# Uses ONLY base Seurat + Matrix (no sceasy / SeuratDisk / reticulate).
# First run the Python exporter to create the plain input files:
#     python benchmark_cellchat_r/export_h5ad_for_cellchat.py
# then this script reads them and writes the Seurat object CellChat expects.
#
# Run from .../QuantumXCT/python/ , or source() it, or:
#     Rscript benchmark_cellchat_r/convert_h5ad_to_seurat.R
suppressPackageStartupMessages({ library(Seurat) })

# locate this script's folder whether sourced or run with Rscript
this_dir <- tryCatch(dirname(normalizePath(sys.frame(1)$ofile)), error = function(e) NA)
if (is.na(this_dir)) {
  a <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", a[grep("^--file=", a)])
  this_dir <- if (length(f)) dirname(normalizePath(f)) else file.path(getwd(), "benchmark_cellchat_r")
}
PYTHON_ROOT <- normalizePath(file.path(this_dir, ".."), mustWork = FALSE)

mtx      <- file.path(this_dir, "counts.mtx.gz")
features <- file.path(this_dir, "features.tsv.gz")
barcodes <- file.path(this_dir, "barcodes.tsv.gz")
meta_csv <- file.path(this_dir, "meta.csv")
rds      <- file.path(PYTHON_ROOT, "dataset_co_mo2.rds")

need <- c(mtx, features, barcodes, meta_csv)
if (!all(file.exists(need))) {
  stop("Missing exported files. Run first:\n  python benchmark_cellchat_r/export_h5ad_for_cellchat.py\nExpected:\n  ",
       paste(need, collapse = "\n  "))
}

# genes x cells sparse matrix
counts <- Seurat::ReadMtx(mtx = mtx, cells = barcodes, features = features,
                          feature.column = 1, cell.column = 1)

obj <- CreateSeuratObject(counts = counts)

meta <- read.csv(meta_csv, row.names = 1, check.names = FALSE)
meta <- meta[colnames(obj), , drop = FALSE]          # align to cell order
obj  <- AddMetaData(obj, metadata = meta)

# CellChat uses the normalized 'data' slot
obj <- NormalizeData(obj, normalization.method = "LogNormalize", scale.factor = 1e4, verbose = FALSE)

stopifnot(all(c("CellType", "BatchID") %in% colnames(obj@meta.data)))
saveRDS(obj, rds)
cat(sprintf("Saved Seurat object (%d genes x %d cells) -> %s\n",
            nrow(obj), ncol(obj), rds))
