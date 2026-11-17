# data_prep: the ovarian dataset

| File | What it is |
|---|---|
| `build_ovarian.ipynb` | Downloads GEO GSE224333, re-attaches the stored annotations, writes `python/dataset_co_mo2.h5ad` and verifies it |
| `dataset_co_mo2_cell_metadata.csv` | Stored cell annotations (barcode, BatchID, CellType, ClusterID, CellCycle) |
| `dataset_co_mo2_genes.txt` | Stored gene order |
| `reference_histograms.csv` | Check values: cells, total counts and panel joint histograms per group |
| `convert_h5ad_rds.R` | h5ad <-> rds conversion for the R tools (CellChat) |

Run `build_ovarian.ipynb` (needs internet and `pip install GEOparse`). Its first step lists
every GEO sample and the group it is assigned to; if one is wrong, set `MAP_OVERRIDES` in the
notebook. The rebuilt file is identical to the one used in the paper, and the last step checks
it against `reference_histograms.csv`.

Why the annotations are stored here: the published object uses plain 10x barcodes with no
sample suffix, so the same barcode can occur in two samples. Each group (cancer / fibroblast x
mono / co-culture) is joined only with its own sample, and cells keep the stored order.

For the R tools:

    Rscript data_prep/convert_h5ad_rds.R h5ad2rds dataset_co_mo2.h5ad dataset_co_mo2.rds

The synthetic dataset is built by `simCellsSpace/simulate_cells.ipynb`; `00_build_datasets.ipynb`
in `python/` lists both and checks they are present.
