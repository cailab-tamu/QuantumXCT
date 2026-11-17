# QuantumXCT — Python package and analyses

This folder holds the `qxct` package, the notebooks that produce every result in the paper,
the scripts that rebuild the input data, and the benchmarks.

## Install

```bash
conda env create -f environment.yml     # Python 3.11, Qiskit 2.1, scanpy, ... -> env "qiskit-env"
conda activate qiskit-env
pip install -e .                        # editable install of the qxct package
pip install GEOparse                    # only to download the ovarian data (00_build_datasets)
```

Run notebooks and scripts from this `python/` folder. The notebooks in subfolders switch to
it automatically, so outputs always land in `python/results/`.

## The package (`qxct/`)

| Module | Contents |
|---|---|
| `model` | `QuantumXct`, the object-oriented driver: encode → screen → search → optimize → analyze |
| `config` | `RunConfig` and its parts: `EncodingConfig`, `BackendConfig`, `SearchConfig`, `OptimizerConfig`, `MitigationConfig` |
| `encoding` | Binary gene states → joint count histograms |
| `circuits` | State preparation, CRX circuits, KL scoring, the three search algorithms, angle optimization |
| `exact` | Exact (state-vector, shot-noise-free) scoring and contribution tables |
| `analysis` | Scope comparison (GRN / LR / both), ablation and contribution tables, L1-vs-L2 comparison |
| `baselines` | Classical and tensor-network baselines (optimal transport, TN transform, Born machine) |
| `scopes` | Candidate scopes: within-cell-type (GRN), cross-cell-type (LR), all |
| `metrics` | Edge and direction recovery against a ground truth **you** provide |
| `hardware` | Backend selection, error mitigation, running the optimized circuit on IBM hardware |
| `plotting` | Histograms, state occupancy, angle traces, interaction networks, circuit figures |

Dataset-specific helpers live outside the package on purpose:
`LR_databases/lr_pairs.py` (known ligand-receptor pairs) and
`simCellsSpace/ground_truth.py` (the synthetic simulator's answer key).

## Use it on your own data

You need two cell types measured in two conditions: alone (mono-culture, the baseline) and
together (co-culture, the interacting state), plus a few genes per cell type.

```python
import numpy as np
from qxct import QuantumXct, RunConfig, BackendConfig, SearchConfig
from qxct.encoding import create_joint_histogram

def counts(X):                       # cells x genes expression matrix -> joint on/off histogram
    return dict(create_joint_histogram((X > 0).astype(int))[0])

genes_ct1 = ["PDGFB", "STAT3", "HES4", "TGFB1"]     # cell type 1 -> qubits 0..3
genes_ct2 = ["PDGFRB", "TGFBR1", "TGFBR2"]          # cell type 2 -> qubits 4..6

config = RunConfig(backend=BackendConfig(shots=5000),
                   search=SearchConfig(algorithm="multi_epoch", threshold=0.005))
qx = QuantumXct(config, gene_list=genes_ct1 + genes_ct2)
qx.encode(counts(X_ct1_mono), counts(X_ct2_mono), counts(X_ct1_co), counts(X_ct2_co))
result = qx.run()                    # screen -> search -> optimize -> analyze

result.topology                      # [(control_qubit, target_qubit), ...] = directed gene pairs
result.angles                        # optimized CRX angles
result.contribution_df               # how much of the change each gate explains
qx.save("results/my_run")
```

### Main settings

| Setting | Default | Meaning |
|---|---|---|
| `EncodingConfig.norm` | `"l1"` | `"l1"`: probabilities equal observed frequencies. `"l2"`: squared counts (emphasizes common states) |
| `SearchConfig.algorithm` | `"multi_epoch"` | `"iterative_n_wise"` (Alg. 1), `"multi_epoch"` (Alg. 2), `"vqe_hybrid"` (Alg. 3, QUBO + VQE) |
| `SearchConfig.candidate_scope` | `"all"` | `"intra"` within cell types (GRN), `"inter"` across cell types (LR), `"all"` both |
| `SearchConfig.threshold` | `0.1` | density-matrix screen; the paper uses `0.005` (keeps all pairs on small panels) |
| `SearchConfig.kl_tol` | `0.05` | minimum KL improvement to accept a gate (smaller = more gates) |
| `SearchConfig.vqe_optimization_level` | `None` (auto) | Qiskit transpiler level for the VQE ansatz: automatically 0 on an ideal simulator and 3 on a real device (or a device emulation); set 0-3 to force |
| `BackendConfig.backend` / `.shots` | local simulator / `1000` | any Qiskit backend; the paper uses 5000 shots |
| `OptimizerConfig.method` / `.initial_angle` | `"COBYLA"` / `π/2` | angle optimizer and its starting angle |

To score without shot noise (exact state vectors), call `qxct.use_exact_scoring(True)`
before searching.

### Mark known ligand-receptor pairs (optional)

QuantumXCT does not need a database. To label which discovered cross-cell edges are
already known LR pairs in the plots:

```python
from LR_databases.lr_pairs import curate_lr_pairs
lr = curate_lr_pairs(genes_ct1 + genes_ct2)     # [('PDGFB', 'PDGFRB'), ('TGFB1', 'TGFBR1'), ...]
qx.plot_report(genes_ct1, genes_ct2, lr, out_dir="results/my_run")
```

The table is the union of CellChatDB (Jin *et al.* 2021) and the LIANA consensus resource
(Dimitrov *et al.* 2022). **If you use it, cite both**; see
[`LR_databases/README.md`](LR_databases/README.md).

### Check recovery against a known network (optional)

```python
from qxct.metrics import evaluate_edge_recovery
rec = evaluate_edge_recovery(result.topology, genes_ct1 + genes_ct2,
                             ground_truth_edges=[("PDGFB", "PDGFRB"), ...])
rec["summary"]                       # precision, recall, direction accuracy, ROC-AUC
```

For the synthetic benchmark the answer key is
`simCellsSpace.ground_truth.synthetic_ground_truth()`.

### Run on quantum hardware

```python
from qxct import MitigationConfig
report = qx.assert_on_hardware(backend=my_ibm_backend,
                               mitigation=MitigationConfig(measurement_mitigation=True))
```

## Reproduce the paper

```bash
jupyter nbconvert --to notebook --execute --inplace 00_build_datasets.ipynb   # data
python run_all.py            # every notebook, analysis and benchmark (long)
python run_all.py --list     # the steps; run a subset with --only KEY ...
```

## Folder layout

Notebooks are the drivers, the `qxct` package holds the code, and every output goes to `results/`.

```
python/
├── 00_build_datasets.ipynb                       build + check both input datasets   (run first)
├── qxct_pred_classic_sim_oo.ipynb                synthetic benchmark                 (main result)
├── qxct_ovarian_cocultured_oo.ipynb              ovarian co-culture, full analysis   (main result)
├── qxct_pred_sc_cocultured_monocultured_oo.ipynb ovarian, single-pipeline version
├── run_all.py                                    runs everything in order (master script)
│
├── qxct/              the package (pip install -e .)
├── data_prep/         scripts that rebuild the data from public sources
├── simCellsSpace/     synthetic-tissue simulator notebook + its ground truth
├── LR_databases/      known ligand-receptor pairs (Jin 2021 + Dimitrov 2022)
├── examples/          small examples that need no dataset
├── benchmarks/        self-contained benchmarks (DM strategies, scaling, hardware)
├── analysis_tools/    scripts behind the supplementary tables and figures
├── cci_benchmarks/    comparison with CellChat, LIANA+ and scTenifoldXct (own environments)
├── tests/             integrity tests (pytest)
└── results/           ALL outputs (created on demand, not in git)
```

Where results go:

```
results/
├── data/                 joint-histogram grids of both datasets (notebook 00)
├── classic_sim/          synthetic notebook
├── ovarian_cocultured/   ovarian notebook (run.json is used by benchmarks/hardware)
├── sc_cocultured/        single-pipeline ovarian notebook
├── analyses/             supplementary tables (csv) and figures (figures/, search_panels/, lr_networks/)
├── scaling/  dm_strategies/  hardware/  ccc_comparison/
├── benchmark_grn_lr/     examples/benchmark_grn_lr.ipynb
└── example_run*          examples/run_pipeline.ipynb (json, ablation csv, figures)
```

`*.h5ad`, `*.rds`, `results/` and caches are not in git: rebuild the data with
`00_build_datasets.ipynb` and the results with the notebooks or `run_all.py`.

## Tests

```bash
python -m pytest tests -q
```
