"""Load the two datasets exactly as the *_oo notebooks do and return count histograms."""
import numpy as np
import scanpy as sc
from scipy.sparse import issparse
from qxct.encoding import create_joint_histogram

SIM_CT1 = ["Gene_50", "Gene_90"]
SIM_CT2 = ["Gene_60", "Gene_70", "Gene_71", "Gene_80"]
OV_CT1 = ["PDGFB", "STAT3", "HES4", "TGFB1"]
OV_CT2 = ["PDGFRB", "TGFBR1", "TGFBR2"]


def _mat(a):
    return a.X.toarray() if issparse(a.X) else np.asarray(a.X)


def _hist(X):
    return dict(create_joint_histogram((X > 0).astype(int))[0])


def load_sim(path="simCellsSpace/sim_cells_datasets_co_mo_simulated_classical.h5ad",
             ct1=SIM_CT1, ct2=SIM_CT2, return_matrices=False):
    ad = sc.read_h5ad(path)
    ad.X = ad.layers["counts"]
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    ct = ad.obs["cell_type"]
    X = {
        "ct1_mo": _mat(ad[ct == "Type_A_Lone", ct1]), "ct2_mo": _mat(ad[ct == "Type_B_Lone", ct2]),
        "ct1_co": _mat(ad[ct == "Type_A_Interact", ct1]), "ct2_co": _mat(ad[ct == "Type_B_Interact", ct2]),
    }
    H = {k: _hist(v) for k, v in X.items()}
    return (H, X) if return_matrices else H


def load_ovarian(path="dataset_co_mo2.h5ad", ct1=OV_CT1, ct2=OV_CT2, return_matrices=False):
    ad = sc.read_h5ad(path)
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    b = ad.obs["BatchID"]
    X = {
        "ct1_mo": _mat(ad[b == "Cancer Cells (Mo)", ct1]), "ct2_mo": _mat(ad[b == "Fibroblasts (Mo)", ct2]),
        "ct1_co": _mat(ad[b == "Cancer Cells (Co)", ct1]), "ct2_co": _mat(ad[b == "Fibroblasts (Co)", ct2]),
    }
    H = {k: _hist(v) for k, v in X.items()}
    return (H, X) if return_matrices else H
