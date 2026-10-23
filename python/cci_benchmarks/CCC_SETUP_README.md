# CCC method comparison — setup & run order

Compare QuantumXCT against three external cell–cell-communication (CCC) tools
on the ovarian dataset: **LIANA** (Python), **scTenifoldXct** (Python), and
**CellChat** (R). Each tool has heavy, conflicting dependencies, so each lives in
its own environment. `qiskit-env` is left untouched.

All tool outputs and the final comparison land in one shared folder:
`QuantumXCT/python/results/ccc_comparison/`.

## 0. Data
- Python tools read `QuantumXCT/python/dataset_co_mo2.h5ad`.
- CellChat (R) reads a Seurat object `dataset_co_mo2.rds`. If you only have the
  `.h5ad`, convert once (see step 4).

## 1. LIANA
```
conda env create -f environment_ccc_liana.yml
conda activate ccc-liana
python benchmark_LIANA_py/liana_benchmark.py      # -> results/ccc_comparison/liana_{Mo,Co}_all.csv
conda deactivate
```

## 2. scTenifoldXct
```
conda env create -f environment_ccc_sctenifoldxct.yml
conda activate ccc-sctenifold
python benchmark_sctenifoldXCT_py/sctenifoldXCT_benchmark.py   # -> results/ccc_comparison/sctenifoldxct_co_all.csv
conda deactivate
```
If the conda solve is slow, instead: `conda create -n ccc-sctenifold python=3.10 pip`,
activate, then `pip install scTenifoldXct scanpy` and let pip resolve.

## 3. CellChat (R)
Install R (>=4.2) and:
```r
install.packages(c("Seurat","dplyr","stringr","remotes","BiocManager"))
BiocManager::install(c("ComplexHeatmap"))
remotes::install_github("jinworks/CellChat")
```
Then run `benchmark_cellchat_r/cellchat_benchmark.R` (from that folder or with
`PYTHON_ROOT` set). It writes `results/ccc_comparison/cellchat_Co_all.csv`
(columns: source, target, ligand, receptor, prob, ...).

## 4. (only if you lack dataset_co_mo2.rds) convert h5ad -> Seurat
From `QuantumXCT/python/`, in R:
```r
# needs sceasy + reticulate (pointing at a python with anndata, e.g. the LIANA env),
# OR SeuratDisk
source("benchmark_cellchat_r/convert_h5ad_to_seurat.R")
```

## 5. QuantumXCT LR scores
Run the ovarian benchmark notebook (`benchmarks_ovarian_cancer.ipynb`) so that
`results/benchmark_ovarian/inter_isolated_contribution.csv` exists. The collector
reads QuantumXCT's LR contributions from it automatically.

## 6. Collect + plot (qiskit-env is fine here)
```
conda activate qiskit-env
python collector_ccc_comparison.py
```
Produces in `results/ccc_comparison/`:
- `ccc_comparison_table.csv` — one row per validated LR pair, one column per method
- `ccc_comparison_panels.png` — per-method score bars
- `ccc_comparison_heatmap.png` — which methods detected each pair

The collector tolerates missing tools: any method you skipped shows up as NaN /
"not detected" rather than failing.
