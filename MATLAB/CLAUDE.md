# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

QuantumXCT is a research project that uses quantum circuits as a proxy for gene-gene interaction networks between cells. The core idea: encode single-cell gene expression patterns as quantum states, apply parameterized entanglement (CX) gates to simulate cell-cell interactions, and search over all possible gate configurations to find the circuit topology that best reproduces observed co-culture expression patterns.

## MATLAB Toolbox Requirements

- **MATLAB Quantum Computing Toolbox** is required (R2023b or later). Key functions used: `quantumCircuit`, `initGate`, `cxGate`, `hGate`, `compositeGate`, `simulate`, `querystates`.
- **Statistics and Machine Learning Toolbox** for PC algorithm causal inference.

There are no build steps, lint commands, or test runners. Scripts are run directly in the MATLAB IDE or via `run('scriptname.m')`.

## Architecture Overview

### Core Computation Pipeline

The main optimization workflow lives in `qmXctSearch.m` / `qmXctBidirectional.m`:

1. **Data ingestion** — `hlp.getn()` extracts binary gene expression patterns from an `sce` struct (`.X` genes×cells, `.g` gene names, `.c_batch_id`, `.c_cell_id`) and converts them to probability distributions.
2. **Quantum state initialization** — gene expression probability amplitudes become initial quantum state amplitudes via `initGate()`, wrapped with `compositeGate()` to map onto specific qubit subsets.
3. **Interaction topology enumeration** — `hlp.linkMatrix_dir_k(K, bag1, bag2)` generates all possible K-link directed configurations between two qubit sets (one per cell type).
4. **Circuit simulation** — for each topology, CX gates are added to the base circuit, simulated, and final state probabilities extracted via `querystates()`.
5. **Scoring** — `hlp.i_kldiverg()` computes symmetrized KL divergence (scaled ×10) between simulated and observed co-culture patterns. Lower score = better topology match.
6. **Visualization** — `hlp.fun_drawreshisto()` renders a 2×2 panel: quantum circuit diagram, link labels, and pattern histograms for both cell types.

`qmXctBidirectional.m` extends the baseline by adding explicit co-culture initial states for bidirectional interaction analysis (four KL terms: forward+backward for each cell type).

### Key Data Structures

| Variable | Description |
|---|---|
| `sce` | Single-cell experiment struct: `.X` (genes×cells), `.g` (gene names), `.c_batch_id`, `.c_cell_id` |
| `pt_f_mo`, `pt_c_mo` | Monoculture probability distributions for cell types F (fibroblast) and C (cancer) |
| `pt_f_co`, `pt_c_co` | Co-culture probability distributions |
| `configsK` | Cell array of K-link configurations; each entry is a 2-column `[source_qubit, target_qubit]` matrix |
| `Y` | Vector of KL divergence scores, one per configuration in `configsK` |
| `targettoplinkers` | Ground-truth target topology, e.g. `[7 2; 1 6; 3 5]` for known interactions |

### Qubit Layout Convention

- Qubits 1–3: fibroblast cell type genes
- Qubits 4–7: cancer cell type genes
- 24 total directed inter-cellular links (3×4 directions × 2 ways)
- A fixed intracellular link CX(4,5) within cancer cells is added as an extra gate in `08_run_qmXct_full.m` / `09_run_qmXct_minimal.m`

### Helper Namespace (`+hlp/`)

All shared utilities live in the `+hlp` MATLAB package and are called as `hlp.functionname()`:

| File | Purpose |
|---|---|
| `getn.m` | Extracts gene expression subsets from `sce`, computes observed pattern frequencies, plots observed vs. theoretical independent distribution |
| `linkMatrix_dir_k.m` | Generates all C(24,K) directed K-link configurations between two qubit bags using `nchoosek` |
| `i_kldiverg.m` | Symmetrized KL divergence: `10 × mean([KL_forward, KL_reverse])`. The ×10 scaling is intentional. |
| `i_obj.m` | Objective function for gradient-based optimization — sets CRY/CRX gate angles, simulates, returns summed KL cost |
| `e_patnfreq.m` | Theoretical expected pattern frequencies assuming gene independence (bag model over all 2^m patterns) |
| `fun_drawreshisto.m` | 4-panel visualization: circuit diagram, link list (with `*` for target links), histograms for both cell types comparing monoculture/co-culture/predicted |
| `permn.m` | Memory-safe permutation-with-repetition generator; used for enumerating binary patterns |

### Experiment Scripts (chronological research progression)

| File | What changed / what was tested |
|---|---|
| `01_k3_cx_kl_oneway.m` | K=3, one-way KL only (mono→co). Baseline pipeline. Adds fixed CX(4,5) intracellular link. |
| `02_k3_cx_kl_bidirectional.m` | K=3, adds bidirectional KL (forward + reverse circuits). Tests if reverse direction helps. |
| `03_k3_cry_optimize_allconfigs.m` | Replaces fixed CX with parameterized **CRY gates**; uses `fminunc` to optimize angles across all K=3 configs. |
| `03b_k3_cry_optimize_allconfigs_piinit.m` | Same as 03 but with π warm-start initial angles. |
| `04_k3_crx_optimize_target.m` | Narrows to target topology only; uses **CRX gates** and 10 independent `fminsearch` trials. |
| `05_k3_integration_test.m` | Incomplete integration test with real data — sets up expression matrices but doesn't finish analysis. |
| `06_k3_cx_gate_order.m` | Tests all 3!=6 orderings of 3 CX gates; confirms non-adjacent CX gates commute (order doesn't matter). |
| `07_k4_topology_search.m` | Extends to **K=4** links; filters configs where all 6 qubits are used; accepts if ≥3/4 links match target. |
| `08_run_qmXct_full.m` | Clean wrapper: calls `qmXctSearch()` with hardcoded fibroblast/cancer genes, target topology, and CX(4,5) extra gate. |
| `09_run_qmXct_minimal.m` | Minimal two-line variant of 08. |

`demo_basic_quantum_gates.m` is a scratchpad demonstrating basic quantum circuit construction (Hadamard, RY, controlled-RY) and statevector measurement.

### Supporting Modules

**`oracle_generation/`**
- `aa.m` — Reference notes (not executable) on variational compilation strategies: inverse trick, swap test, Hadamard test for state-to-state mapping.
- `buildCSWAP.m` — Builds a controlled-SWAP (Fredkin) gate from Clifford+T primitives (H, CNOT, T, T†).
- `swaptest_matlab_example.m` — Working swap-test demo: estimates ⟨ψ|φ⟩² via ancilla + controlled-SWAP, plots simulated vs. analytical cost.

**`interact_simulation/`**
- `reconstructFromHistogram.m` — Inverts histogram counting: converts a 2^m frequency vector back to a binary m×N gene expression matrix (inverse of `hlp.getn`).

**`gen_synthetic_cci_data.m`** — Synthetic data generator: creates ground-truth single-cell matrices with programmed A→B ligand-receptor (L1-R1) interaction. Used for benchmarking.

**`PCAlgorithm.m` / `demo_pc_algorithm.m`** — Complete PC causal discovery algorithm (skeleton learning via conditional independence + edge orientation via v-structures and Meek's rules). Python port in `pc_algrithm_causal_net.py`.

**`untitled.py`** — QuTiP Bloch sphere animation of RY rotation on |0⟩, |1⟩, |+⟩, |-⟩ states, saved as GIF.

## Typical Usage

```matlab
% Load data (sce struct with .X, .g, .c_batch_id, .c_cell_id fields)
load('sevengenes_test.mat')

% Run quantum interaction search (K=3 links, two cell type gene sets)
[Y, configsK] = qmXctSearch(sce, genes1, genes2, tags, target_links, 3)

% Find best configuration
[~, idx] = min(Y);
best_config = configsK{idx};

% Visualize a specific configuration
hlp.fun_drawreshisto(idx, Cc, configsK, pt_f_mo, pt_c_mo, pt_f_co, pt_c_co, genes1, genes2, sv)
```

## Key Scientific Findings Encoded in Scripts

- **Gate order does not matter** for non-adjacent qubits (confirmed by `06_k3_cx_gate_order.m`).
- **Bidirectional KL** (`qmXctBidirectional` / `02`) gives more robust topology identification than one-way.
- **CRY/CRX parameterization** (`03`/`04`) allows continuous optimization on top of the discrete topology search.
- **K=4** (`07_k4_topology_search.m`) can recover K=3 ground truth if 3 of 4 links match, offering redundancy tolerance.
