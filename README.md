# QuantumXCT

**Learning interaction-induced state transformations in cell-cell communication with quantum circuits.**

QuantumXCT infers cell-cell communication (CCC) from single-cell RNA-seq without relying on a
ligand-receptor database. It encodes the gene-activity states of two cell types as quantum
amplitudes and learns a sparse circuit of controlled rotations (CRX gates) that turns the
*non-interacting* state (mono-culture) into the *interacting* state (co-culture). Each gate
is a named, directed, activity-gated interaction between two genes, so the learned circuit
reads directly as an interaction network, and every gate's contribution to the state change
can be measured.

> Romero S, Gupta S, Chapkin RS, Cai JJ. *QuantumXCT: Learning Interaction-Induced State
> Transformation in Cell-Cell Communication via Quantum Entanglement and Generative Modeling.*
> Under revision, *Machine Learning: Science and Technology* (MLST-105562).

---

## What it does

1. **Encode.** Binarize the selected genes per cell, count the joint on/off states of each cell
   type, and turn the counts into amplitudes (square-root "L1" encoding by default, so the
   measured probabilities equal the observed frequencies).
2. **Search.** Screen candidate gene pairs with a density-matrix heuristic, then find the gate
   topology that best maps mono-culture to co-culture. Three search algorithms: N-wise local
   search, multi-epoch greedy, and a QUBO + VQE hybrid.
3. **Optimize and read out.** Tune the gate angles (COBYLA), then rank each gate by how much of
   the state change it explains (exact, shot-noise-free contribution analysis).

It runs on a local simulator by default and on IBM quantum hardware when a backend is given.

## Quick start

```bash
git clone https://github.com/cailab-tamu/QuantumXCT.git
cd QuantumXCT/python
conda env create -f environment.yml      # creates "qiskit-env"
conda activate qiskit-env
pip install -e .                         # installs the qxct package
```

```python
from qxct import QuantumXct, RunConfig
qx = QuantumXct(RunConfig(), gene_list=genes)          # genes: cell type 1 then cell type 2
qx.encode(ct1_mono, ct2_mono, ct1_co, ct2_co)          # joint-state count histograms
result = qx.run()                                      # screen -> search -> optimize
```

## Reproduce the paper

No data are stored in the repository; everything is rebuilt from public sources.

Follow the step-by-step **[walkthrough in `python/README.md`](python/README.md#walkthrough-reproduce-the-paper)**:
it lists every notebook in the order to run it, what it does and where its output goes.

1. Data: `python/simCellsSpace/simulate_cells.ipynb` (synthetic tissue) and
   `python/data_prep/build_ovarian.ipynb` (ovarian cancer-fibroblast co-culture, GEO **GSE224333**),
   each verified against stored reference histograms.
2. Main results: `python/qxct_pred_classic_sim_oo.ipynb` and `python/qxct_ovarian_cocultured_oo.ipynb`.
3. Benchmarks: `python/benchmarks/{dm_strategies,scaling,hardware}/`.
4. Supplementary tables and figures: `python/analysis_tools/`, or everything at once with
   `python run_all.py` (from `python/`).

All results are written to `python/results/`.

## Where to go next

| Read | For |
|---|---|
| [`python/README.md`](python/README.md) | Installing and using the `qxct` package, folder layout, where results go |
| [`python/examples/`](python/examples/) | Small runnable examples that need no dataset |

## Repository layout

```
QuantumXCT/
├── python/        the qxct package, notebooks, data builders, benchmarks  (start here)
├── MATLAB/        the original MATLAB implementation
└── manuscript/    source datasets and early manuscript material
```

## Citation

If you use QuantumXCT, please cite the paper above. If you use the bundled ligand-receptor
tables (`python/LR_databases/`), please also cite their sources:

- Jin S *et al.* Inference and analysis of cell-cell communication using CellChat.
  *Nat Commun* **12**, 1088 (2021). https://doi.org/10.1038/s41467-021-21246-9
- Dimitrov D *et al.* Comparison of methods and resources for cell-cell communication
  inference from single-cell RNA-Seq data. *Nat Commun* **13**, 3224 (2022).
  https://doi.org/10.1038/s41467-022-30755-0

## License

MIT. See [`LICENSE`](LICENSE).
