# QuantumXCT — Object-Oriented API

This folder showcases the object-oriented interface to `qxct`. Instead of calling
a dozen free functions and threading the same long list of keyword arguments
through each one, you configure a run once with a few dataclasses and drive the
pipeline through a single `QuantumXct` object.

The original functional API in `qxct.circuits` is unchanged and still fully
supported — the OO layer is a thin, config-driven wrapper on top of it.

## Files

| File | What it does |
|------|--------------|
| `run_pipeline.ipynb` | End-to-end pipeline on synthetic data: encode → screen → search → optimize → analyze → hardware validation → save. |
| `benchmark_grn_lr.ipynb` | Benchmarks the three search scopes (GRN+LR, GRN-only, LR-only) and writes the KL-reduction, cumulative-ablation and isolated-contribution tables. |

Both run without any dataset or real hardware, from any folder: their first cell switches to `python/`, so outputs go to `python/results/`. Replace the `synthetic_data()` /
`load_counts()` helpers with your own joint histograms to use them for real.

## Quick start

```python
import numpy as np
from qxct import QuantumXct, RunConfig, EncodingConfig, SearchConfig, BackendConfig

config = RunConfig(
    encoding=EncodingConfig(norm="l1"),                 # L1 is the default encoding
    backend=BackendConfig(backend=None, shots=2000),    # None -> local AerSimulator
    search=SearchConfig(candidate_scope="all"),         # GRN + LR
)

qx = QuantumXct(config, gene_list=genes)
qx.encode(ct1_basal, ct2_basal, ct1_interacting, ct2_interacting)
result = qx.run()          # -> result.topology, result.angles, result.kl_sum, result.contribution_df
```

## The configuration objects

Everything is grouped into `RunConfig`, which holds five sub-configs. Every field
has a default, so `RunConfig()` alone is a valid run.

### `EncodingConfig` — how histograms become quantum states
| Field | Default | Meaning |
|-------|---------|---------|
| `norm` | `"l1"` | `"l1"`: amplitude = √(count / Σcount), so measurement probability ∝ frequency (frequency-preserving, **default**). `"l2"`: amplitude = count / ‖count‖₂, probability ∝ count² (original paper encoding). |

The chosen convention is applied **once**, consistently, to the initial-state
circuits, the target distributions and the density-matrix screening — so the KL
comparison is always between two states in the same encoding. This is the fix for
the previous behaviour where the L1/L2 choice was not threaded through every
stage.

### `BackendConfig` — where circuits run
| Field | Default | Meaning |
|-------|---------|---------|
| `backend` | `None` | `None` → local `AerSimulator`; otherwise any Qiskit backend / IBM session. |
| `shots` | `1000` | Shots per circuit evaluation during the search. |
| `optimization_level` | `1` | Transpiler preset level while scoring. |

### `SearchConfig` — candidate screening + topology search
| Field | Default | Meaning |
|-------|---------|---------|
| `algorithm` | `"multi_epoch"` | `"multi_epoch"` or `"iterative_n_wise"`. |
| `candidate_scope` | `"all"` | `"all"` = GRN + LR; `"intra"` = GRN (within-cell) only; `"inter"` = LR (cross-cell) only. |
| `threshold` | `0.1` | Density-matrix-difference screening threshold. |
| `search_mode` | `"pos"` | `"pos"`, `"neg"` or `"abs"`. |
| `kl_tol` | `0.05` | Occam's-razor significance threshold. |
| `n_epochs` | `10` | Greedy-search restarts. |
| `max_greedy_depth` | `30` | Max gates added per greedy path (a.k.a. "max iters"). |
| `nchoose` | `2` | Pair size for the n-wise search. |

### `OptimizerConfig` — CRX angle refinement
| Field | Default | Meaning |
|-------|---------|---------|
| `method` | `"COBYLA"` | SciPy optimizer. |
| `initial_angle` | `π/2` | Starting angle for every CRX (matches the CRX(π/2) topology default). |
| `etol` | `1e-6` | Optimizer tolerance. |

### `MitigationConfig` — error mitigation for the hardware run
Each toggle is independent, so you enable only the ones you want. On a local
simulator they are harmless no-ops; on real IBM hardware they map onto
`SamplerV2` options.

| Field | Default | Meaning |
|-------|---------|---------|
| `dynamical_decoupling` | `False` | Insert DD sequences on idle qubits. |
| `twirling_gates` | `False` | Pauli-twirl gates. |
| `twirling_measure` | `False` | Twirl measurements. |
| `measurement_mitigation` | `False` | Readout / TREX measurement-error mitigation. |
| `resilience_level` | `None` | Resilience level (if the primitive exposes it). |
| `optimization_level` | `3` | Transpiler level for the hardware circuit. |
| `shots` | `4000` | Shots for the validation run. |

`MitigationConfig.all_on()` returns a config with every option enabled.

## Validating on real hardware

The search and optimization run on a simulator. Once you have the best topology
and angles, run that single circuit **once** on a real device to confirm the
result survives noise:

```python
from qiskit_ibm_runtime import QiskitRuntimeService
backend = QiskitRuntimeService().least_busy(operational=True, simulator=False)

hw = qx.assert_on_hardware(backend=backend)   # uses config.mitigation
print(hw["kl_total"], hw["backend"], hw["mitigation"])
```

`assert_on_hardware` rebuilds the optimized circuit, transpiles it to the backend,
applies the mitigation toggles and returns the measured counts plus the KL
divergence against the target distributions, so you can compare hardware vs. the
ideal simulation.

## The three benchmark scopes

`benchmark_grn_lr.ipynb` (via `qxct.analysis.compare_scopes`) runs the full
search+optimize for each scope and reports:

* **GRN + LR** (`scope="all"`) — search every candidate link.
* **GRN only** (`scope="intra"`) — within-cell links, optimized one cell at a
  time (cell 1 to convergence, then cell 2); cross-cell LR links cannot contribute.
* **LR only** (`scope="inter"`) — cross-cell (ligand-receptor) links **only**:
  *every* cross-register pair (both directions), with **no density-matrix
  screening** and **no GRN (intra-cell) links** at all.

For each scope it produces two contribution tables for the final topology:

* **Cumulative ablation reduction** — add the gates back in order; running KL and
  cumulative reduction from baseline (order-dependent).
* **Isolated / leave-one-out contribution** — each gate's isolated effect (that
  gate alone) and its marginal effect (full circuit minus that gate);
  order-independent importance.
