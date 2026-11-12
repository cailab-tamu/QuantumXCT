"""Regenerate the synthetic benchmark dataset (reproduction step).

One-time DATA-CONSTRUCTION step, run by hand BEFORE the benchmark; NOT part of
run_all.py. It writes ``simCellsSpace/sim_cells_datasets_co_mo_simulated_classical.h5ad``
in the shape ``analysis_tools/data.load_sim`` expects: raw counts in
``layers['counts']``, ``obs['cell_type']`` in {Type_A_Lone, Type_B_Lone,
Type_A_Interact, Type_B_Interact}, genes named ``Gene_0 .. Gene_119``.

The generator is the same model as ``simCellsSpace/simulate_cells.py`` (a spatial
LR -> receptor -> GRN -> feedback simulation sampled with a negative-binomial count
model), reproduced here headless (no plotting) and SEEDED so it is deterministic.

Reproducibility note: the original notebook seeded only the spatial layout
(make_blobs, random_state=42); the count sampling and dropout used un-seeded
numpy, so the committed h5ad cannot be reproduced bit-for-bit. This script seeds
everything (``--seed``, default 42), so it is reproducible going forward. To
reproduce the published numbers exactly, keep the committed h5ad; to regenerate
from scratch, run this and then re-run the pipeline so all downstream numbers come
from the same seeded dataset.

Run from QuantumXCT/python:
    python data_prep/prepare_sim.py
    python data_prep/prepare_sim.py --seed 7 --out simCellsSpace/sim_custom.h5ad

Requires: numpy, scanpy, anndata, scipy, scikit-learn.
"""
from __future__ import annotations
import argparse
import os

import numpy as np

# published parameters (identical to the EXECUTION block of simCellsSpace/simulate_cells.py)
PARAMS = dict(
    n_cells=4000,
    n_genes=120,
    lr_forward={50: 60},
    lr_retro={80: 90},
    grn_logic={
        70: {71: 1.5, 72: -1.5, 80: 2.0},   # spatial / reactive module (via receptor 60 -> master 70)
        73: {74: 1.5, 75: 2.0, 76: -1.5},   # autonomous / constitutive module
    },
    receptor_to_grn={60: 70},
    sparsity_factor=0.4,
)


def simulate_complex_benchmarking_tissue(n_cells, n_genes, lr_forward, lr_retro,
                                         grn_logic, receptor_to_grn, sparsity_factor):
    """Spatial LR/GRN tissue simulation -> AnnData of raw counts (see module docstring)."""
    import anndata as ad
    from scipy.spatial.distance import cdist
    from sklearn.datasets import make_blobs

    centroids = [[10, 10], [12, 12], [50, 10], [10, 50]]
    coords, labels = make_blobs(n_samples=n_cells, centers=centroids, cluster_std=2.0, random_state=42)
    role_map = {0: "Type_A_Interact", 1: "Type_B_Interact", 2: "Type_A_Lone", 3: "Type_B_Lone"}
    cell_types = np.array([role_map[l] for l in labels])

    mu_0 = -2.0
    latent_X = np.random.normal(mu_0, 0.5, (n_cells, n_genes))

    A_mat = np.zeros((n_genes, n_genes))
    if grn_logic:
        for source, targets in grn_logic.items():
            for target, weight in targets.items():
                A_mat[target, source] = weight

    dist_mat = cdist(coords, coords)
    sigma = 4.0
    W = np.exp(-(dist_mat ** 2) / (2 * sigma ** 2))
    np.fill_diagonal(W, 0)

    is_a = np.char.startswith(cell_types, "Type_A")
    is_b = np.char.startswith(cell_types, "Type_B")

    for l_idx in lr_forward.keys():
        latent_X[is_a, l_idx] += 3.5

    for i in range(n_cells):
        if "Type_B" in cell_types[i]:
            for l_idx, r_idx in lr_forward.items():
                signal = np.sum(W[i, is_a] * np.exp(latent_X[is_a, l_idx]))
                latent_X[i, r_idx] += np.log1p(signal * 2.0)
                if r_idx in receptor_to_grn and latent_X[i, r_idx] > 0.5:
                    ms = receptor_to_grn[r_idx]
                    latent_X[i, ms] += 2.5

    for i in range(n_cells):
        current_snapshot = latent_X[i, :].copy()
        for target_gene in range(n_genes):
            sources = np.where(A_mat[target_gene, :] != 0)[0]
            for source_gene in sources:
                weight = A_mat[target_gene, source_gene]
                if current_snapshot[source_gene] > -1.0:
                    delta = (current_snapshot[source_gene] - mu_0)
                    latent_X[i, target_gene] += weight * delta

    for i in range(n_cells):
        if "Type_A" in cell_types[i]:
            for l_idx, r_idx in lr_retro.items():
                signal = np.sum(W[i, is_b] * np.exp(latent_X[is_b, l_idx]))
                latent_X[i, r_idx] += np.log1p(signal * 2.5)

    latent_X = np.clip(latent_X, -10, 6)
    mu = np.exp(latent_X)
    phi = 0.8
    counts = np.random.negative_binomial(1 / phi, (1 / phi) / (1 / phi + mu))
    dropout_mask = np.random.binomial(1, 1 - sparsity_factor, size=counts.shape)
    counts = counts * dropout_mask

    adata = ad.AnnData(X=counts.astype(np.float32))
    adata.obs["cell_type"] = cell_types
    adata.obsm["spatial"] = coords
    adata.var_names = [f"Gene_{i}" for i in range(n_genes)]
    return adata


def main():
    ap = argparse.ArgumentParser(description="Regenerate the synthetic benchmark h5ad (seeded).")
    ap.add_argument("--out", default=os.path.join("simCellsSpace",
                    "sim_cells_datasets_co_mo_simulated_classical.h5ad"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import scanpy as sc
    import scipy.sparse as sp

    np.random.seed(args.seed)
    adata = simulate_complex_benchmarking_tissue(**PARAMS)

    adata.layers["counts"] = adata.X.copy()      # raw counts (what load_sim reads)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.X = sp.csr_matrix(adata.X)
    adata.layers["counts"] = sp.csr_matrix(adata.layers["counts"])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    adata.write_h5ad(args.out)
    print(f"[prepare_sim] wrote {args.out}: {adata.shape[0]} cells x {adata.shape[1]} genes (seed={args.seed})")
    print("[prepare_sim] cell_type counts:", dict(adata.obs["cell_type"].value_counts()))
    print("[prepare_sim] sanity: run  python -c \"from analysis_tools.data import load_sim; "
          "load_sim(); print('load_sim OK')\"")


if __name__ == "__main__":
    main()
