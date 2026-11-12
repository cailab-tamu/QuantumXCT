#!/usr/bin/env Rscript
# h5ad <-> rds converter (reproduction helper for the R-based CCI tools).
#
# One-time DATA-CONSTRUCTION helper, run by hand; NOT part of run_all.py. It bridges
# the Python AnnData files and the R objects CellChat / LIANA-R expect.
#
# Uses zellkonverter (Bioconductor) for AnnData I/O; the .rds holds a
# SingleCellExperiment by default, or a Seurat object with `seurat`.
#
# Usage (run in the R / CCI conda env, from QuantumXCT/python):
#   Rscript data_prep/convert_h5ad_rds.R h5ad2rds dataset_co_mo2.h5ad dataset_co_mo2.rds          # -> SCE
#   Rscript data_prep/convert_h5ad_rds.R h5ad2rds dataset_co_mo2.h5ad dataset_co_mo2_seurat.rds seurat
#   Rscript data_prep/convert_h5ad_rds.R rds2h5ad dataset_co_mo2.rds dataset_co_mo2_roundtrip.h5ad
#
# Requires: zellkonverter, SingleCellExperiment (Bioconductor); Seurat only if you
# use/produce a Seurat object. Install:
#   if (!requireNamespace("BiocManager", quietly=TRUE)) install.packages("BiocManager")
#   BiocManager::install(c("zellkonverter", "SingleCellExperiment"))
#   install.packages("Seurat")   # optional

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop(paste(
    "Usage:",
    "  Rscript convert_h5ad_rds.R h5ad2rds <in.h5ad> <out.rds> [sce|seurat]",
    "  Rscript convert_h5ad_rds.R rds2h5ad <in.rds> <out.h5ad>",
    sep = "\n"))
}
mode   <- args[[1]]
infile <- args[[2]]
outfile <- args[[3]]
kind   <- if (length(args) >= 4) tolower(args[[4]]) else "sce"

need <- function(pkg) if (!requireNamespace(pkg, quietly = TRUE))
  stop(sprintf("Package '%s' is required. See the header for install instructions.", pkg))

if (mode == "h5ad2rds") {
  need("zellkonverter"); need("SingleCellExperiment")
  message(sprintf("[convert] reading AnnData: %s", infile))
  sce <- zellkonverter::readH5AD(infile, verbose = TRUE)
  if (kind == "seurat") {
    need("Seurat")
    # Seurat needs a counts assay; fall back to X if there is no 'counts'
    if (!"counts" %in% SummarizedExperiment::assayNames(sce)) {
      an <- SummarizedExperiment::assayNames(sce)[1]
      message(sprintf("[convert] no 'counts' assay; using '%s' as counts", an))
      SummarizedExperiment::assay(sce, "counts") <- SummarizedExperiment::assay(sce, an)
    }
    obj <- Seurat::as.Seurat(sce, counts = "counts", data = NULL)
    message("[convert] converted SCE -> Seurat")
  } else {
    obj <- sce
  }
  saveRDS(obj, outfile)
  message(sprintf("[convert] wrote %s (%s)", outfile, kind))

} else if (mode == "rds2h5ad") {
  need("zellkonverter")
  message(sprintf("[convert] reading rds: %s", infile))
  obj <- readRDS(infile)
  if (inherits(obj, "Seurat")) {
    need("Seurat")
    obj <- Seurat::as.SingleCellExperiment(obj)
    message("[convert] converted Seurat -> SCE")
  } else if (!inherits(obj, "SingleCellExperiment")) {
    stop("rds must contain a SingleCellExperiment or a Seurat object.")
  }
  zellkonverter::writeH5AD(obj, outfile, verbose = TRUE)
  message(sprintf("[convert] wrote %s", outfile))

} else {
  stop(sprintf("Unknown mode '%s'. Use 'h5ad2rds' or 'rds2h5ad'.", mode))
}
