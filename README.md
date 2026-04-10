# QuantumXCT

**Hybrid Quantum-Classical Cell-Cell Communication Inference from scRNA-seq Data**

QuantumXCT encodes single-cell gene expression states as quantum amplitudes and learns a parameterized CRX-gate topology that transforms non-interacting (mono-culture) states into interacting (co-culture) states. The discovered topology is decoded back into directional gene regulatory interactions between cell types.

---

## Repository Layout

```
QuantumXCT/
├── python/
│   ├── pyproject.toml                 ← pip-installable package config (run: pip install -e .)
│   ├── qxct/                          ← Python package
│   │   ├── __init__.py
│   │   ├── encoding.py                ← Quantum state encoding (histograms → amplitudes)
│   │   └── circuits.py                ← Circuits, topology search, VQE, analysis
│   │
│   ├── qxct_pred_classic_sim.ipynb         ← Demo: simulated cell data
│   ├── qxct_pred_sc_cocultured_monocultured.ipynb  ← Real scRNA-seq analysis
│   ├── environment.yml                ← Conda environment spec
│   ├── LR_databases/                  ← Ligand-receptor reference databases
│   └── simCellsSpace/                 ← Simulated dataset generation
│
├── MATLAB/                            ← Original MATLAB implementation
├── manuscript/                        ← Datasets and manuscript figures
└── README.md                          ← This file
```

---

## Quickstart

### 1. Create the conda environment

```bash
conda env create -f python/environment.yml
conda activate qiskit-env
```

### 2. Install the qxct package (editable mode)

```bash
cd python
pip install -e .
```

`pyproject.toml` lives here, at the `python/` level, so `pip` can discover the `qxct/` package folder automatically.

Or skip the install step entirely — each notebook adds the `python/` directory to the path automatically via a relative import at the top of the path-setup cell.

### 3. Run a notebook

```bash
cd python
jupyter notebook qxct_pred_classic_sim.ipynb
```

---

## Method Overview

QuantumXCT operates in three stages:

**Stage 1 — Quantum State Encoding** (`qxct.encoding`)

Binarizes scRNA-seq expression matrices and builds joint frequency histograms C(s) for each gene panel and condition (mono-culture / co-culture). Counts are L2-normalized to produce valid quantum state amplitude vectors |ψ⟩.

**Stage 2 — Topology Search** (`qxct.circuits`)

Given mono-culture states |ψ_Mo⟩ as input and co-culture states |ψ_Co⟩ as targets, the algorithm searches for a sequence of CRX(π/2) gates connecting two cell-type qubit registers that minimizes the combined KL divergence L(τ, θ) = D_KL(P_sim ‖ P_target). Three complementary search algorithms are provided:

| Algorithm | Function | Description |
|-----------|----------|-------------|
| **Alg. 1** N-wise Iterative Local Search | `find_best_cnot_sequence_iterative_n_wise` | Iterates single-gate insertion, pairwise addition, and pruning phases |
| **Alg. 2** Multi-Epoch Greedy | `find_best_cnot_sequence_multi_epoch` | Multiple greedy forward-search paths with removal refinement |
| **Alg. 3** QUBO + VQE Hybrid | `build_kl_divergence_matrix_interaction` → `kl_to_qubo_matrix` → `vqe_solver` → `run_vqe_hybrid_search` | Encodes topology selection as a QUBO and solves with VQE/QAOA |

All algorithms use `find_cnot_candidates_from_state_diff` (density matrix Δρ heuristic, Appendix B of the paper) to prune the candidate gate space before searching.

**Stage 3 — Angle Optimization & Decoding** (`qxct.circuits`)

Once the optimal topology τ* is found, CRX rotation angles θ are fine-tuned via COBYLA minimization (`optimize_crx_angles`, initial θ = 0). The final circuit is decoded into a directed gene interaction network via `analyze_and_summarize_network` and visualized with `plot_quantum_relay_network`.

---

## Package Reference

### `qxct.encoding`

| Function | Description |
|----------|-------------|
| `create_joint_histogram(X)` | Raw joint frequency counts C(s) from a boolean expression matrix |
| `create_percent_joint_histogram(X)` | Same, normalized to percentages |
| `count_boolean_vector_occurrences(v)` | Marginal counts for a single gene vector |
| `plot_joint_histogram(counts, n_qubits, ...)` | Bar-chart visualization of a histogram |

### `qxct.circuits` — State Preparation

| Function | Description |
|----------|-------------|
| `build_initial_state_circuit(sparse_dict)` | Creates a Qiskit circuit initialized to the given amplitude dictionary (L2-normalized internally) |
| `vector_normalize_dictionary_values(d)` | L2-normalizes a histogram count dictionary for use as state amplitudes |

### `qxct.circuits` — Circuit Construction

| Function | Description |
|----------|-------------|
| `concatenate_circuits_with_separate_measurements(c1, c2)` | Tensor-products two circuits onto disjoint qubit registers with separate classical registers `c_measure1` / `c_measure2` |
| `apply_entanglement_topology(base, n1, topology)` | Applies CRX(π/2) gates at specified (control, target) qubit pairs — used during topology search |
| `add_crx_gates_and_measurements_to_circuit(base, n1, topology, angles)` | Applies CRX gates with individually specified angles — used during angle optimization |

### `qxct.circuits` — Scoring

| Function | Description |
|----------|-------------|
| `score_circuit_kl_divergences(circuit, target1, target2, nshots)` | Simulates circuit and returns KL divergences against both target distributions |
| `calculate_kl_divergence(p, q)` | KL divergence D_KL(P ‖ Q) with ε-smoothing |
| `get_probability_distribution(counts, n, shots)` | Converts raw measurement counts to a complete probability distribution |

### `qxct.circuits` — Topology Search

| Function | Description |
|----------|-------------|
| `find_cnot_candidates_from_state_diff(...)` | Density matrix Δρ heuristic — prunes candidate gate list (Appendix B) |
| `find_best_cnot_sequence_iterative_n_wise(...)` | **Algorithm 1**: N-wise iterative local search |
| `find_best_cnot_sequence_multi_epoch(...)` | **Algorithm 2**: Multi-epoch greedy search with removal refinement |
| `build_kl_divergence_matrix_interaction(...)` | **Algorithm 3, step 1**: Builds pairwise KL matrix for QUBO formulation |
| `kl_to_qubo_matrix(kl_matrix, baseline_kl)` | **Algorithm 3, step 2**: Converts KL matrix to QUBO coefficients |
| `vqe_solver(ansatz, hamiltonian, backend)` | **Algorithm 3, step 3**: VQE optimization of the Ising Hamiltonian |
| `run_vqe_hybrid_search(...)` | **Algorithm 3, step 4**: Full VQE → greedy hybrid pipeline |

### `qxct.circuits` — Angle Optimization

| Function | Description |
|----------|-------------|
| `optimize_crx_angles(c1, c2, target1, target2, topology, ...)` | COBYLA minimization of CRX rotation angles for a fixed topology. Default initial angle: **0.0 rad** (matches paper, Sec. 2.3.3) |

### `qxct.circuits` — Analysis & Visualization

| Function | Description |
|----------|-------------|
| `plot_measurement_histograms(circuit, nshots)` | Plots `c_measure1` / `c_measure2` histograms side-by-side |
| `analyze_and_summarize_network(c1, c2, topology, angles, ...)` | Ablation study: quantifies per-gate KL contribution; returns DataFrame + hub dict + gene set |
| `plot_quantum_relay_network(topology, genes, ...)` | Biologically-constrained network visualization with L-R pair bridging |

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `qxct_pred_classic_sim.ipynb` | Full pipeline on a simulated scRNA-seq dataset (in-silico validation). Covers all three search algorithms, CRX angle optimization, and VQE/QUBO. |
| `qxct_pred_sc_cocultured_monocultured.ipynb` | Real dataset: macrophage-containing lung organoids (GSM7286480–85). Reproduces paper Figures 3A/3B and Table 2. |

---

## Citation

If you use QuantumXCT in your work, please cite:

> Romero S. et al. *QuantumXCT: A Hybrid Quantum-Classical Framework for Cell-Cell Communication Inference.* (2025).

---

## Environment

```yaml
# python/environment.yml
name: qiskit-env
dependencies:
  - python=3.10
  - numpy=1.26.4
  - scipy
  - scanpy
  - pandas
  - matplotlib
  - scikit-learn
  - qiskit=2.1
  - qiskit-ibm-runtime
  - pip:
    - qiskit_aer
    - qiskit-optimization
```

---

## License

MIT — see `LICENSE` file.
