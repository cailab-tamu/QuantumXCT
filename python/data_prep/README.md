# data_prep — rebuilding the inputs from public data

This folder reconstructs the two datasets the benchmark reads. It is a one-time
reproduction step you run by hand, as a FIRST step, before anything in `run_all.py`.
Nothing here is imported or triggered by `run_all.py`; the benchmark simply assumes
`dataset_co_mo2.h5ad` and `simCellsSpace/sim_cells_datasets_co_mo_simulated_classical.h5ad`
already exist.

Run everything from the `QuantumXCT/python` folder, in the same conda env you use for
the pipeline.

## Files

- `export_metadata.py` — exports the preserved cell annotations from an existing
  annotated `dataset_co_mo2.h5ad` to `dataset_co_mo2_cell_metadata.csv`
  (barcode, BatchID, CellType, ClusterID, CellCycle) and the gene order to
  `dataset_co_mo2_genes.txt`. These two small files are committed with the repo so
  the annotations survive even without the large h5ad; `pull_ovarian.py` re-attaches
  them to freshly downloaded counts. You only re-run this if you change the
  annotated object.
- `pull_ovarian.py` — downloads the raw counts for GEO series GSE224333 (in-vitro
  monoculture + co-culture scRNA-seq of ovarian clear-cell carcinoma cells and
  fibroblasts) and re-attaches the preserved annotations, joining on the cell
  barcode within each sample, to rebuild `dataset_co_mo2.h5ad`.
- `prepare_sim.py` — regenerates the synthetic benchmark dataset with a fixed seed,
  writing `simCellsSpace/sim_cells_datasets_co_mo_simulated_classical.h5ad`.
- `convert_h5ad_rds.R` — converts an h5ad to an `.rds` (SingleCellExperiment or
  Seurat) and back, for the R-based cell-cell-interaction tools (CellChat).

## Order

1. Ovarian counts:

       python data_prep/pull_ovarian.py --dry-run      # inspect GSMs, confirm the BatchID mapping
       python data_prep/pull_ovarian.py                # build dataset_co_mo2.h5ad

   The dry run lists every sample in the series with the BatchID it was matched to
   (Cancer Cells / Fibroblasts x Mono / Co). If a title was classified wrong, pass an
   explicit mapping, e.g. `--map GSM1234="Cancer Cells (Mo)" GSM1235="Fibroblasts (Co)"`.
   Then verify:

       python -c "from analysis_tools.data import load_ovarian; load_ovarian(); print('load_ovarian OK')"

2. Synthetic dataset:

       python data_prep/prepare_sim.py
       python -c "from analysis_tools.data import load_sim; load_sim(); print('load_sim OK')"

3. (Only for the R CCI tools) convert as needed, e.g.

       Rscript data_prep/convert_h5ad_rds.R h5ad2rds dataset_co_mo2.h5ad dataset_co_mo2.rds

## Notes

- The ovarian object uses plain 10x barcodes with no per-sample suffix, so the same
  barcode can appear in more than one sample. `pull_ovarian.py` joins each BatchID
  group only against its own sample's matrix, which keeps the cells distinct.
- `prepare_sim.py` is seeded, so it is reproducible. The originally committed
  synthetic h5ad was generated without a count-level seed and cannot be reproduced
  bit-for-bit; to reproduce the published numbers exactly, keep the committed h5ad,
  and to regenerate from scratch, run `prepare_sim.py` and then re-run the pipeline
  so all downstream numbers come from the same seeded dataset.

## Requirements

- Python: `GEOparse`, `scanpy`, `anndata`, `pandas`, `scipy`, `scikit-learn`
  (`pip install GEOparse scanpy`).
- R (only for the converter): `zellkonverter`, `SingleCellExperiment` from
  Bioconductor, and `Seurat` if you want a Seurat `.rds`.
