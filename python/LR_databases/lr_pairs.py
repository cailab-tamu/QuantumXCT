"""
LR_databases.lr_pairs
=====================
Known ligand-receptor (LR) pairs for annotating QuantumXCT networks.

QuantumXCT does not need an LR database to run: any cross-cell gene pair can be
a candidate interaction. This helper only answers "which discovered cross-cell
edges are already known LR pairs?", so plots can draw them as validated links.

Database: ``Human-merged-Jin-Dimitrov-LR-pairs.csv`` in this folder (columns
``ligand, receptor, source``), the union of the two human resources below. The
two original tables are kept next to it for provenance.

If you use these tables, please cite both sources:
  * Jin S, Guerrero-Juarez CF, Zhang L, et al. Inference and analysis of
    cell-cell communication using CellChat. Nat Commun 12, 1088 (2021).
    https://doi.org/10.1038/s41467-021-21246-9
  * Dimitrov D, Türei D, Garrido-Rodriguez M, et al. Comparison of methods and
    resources for cell-cell communication inference from single-cell RNA-Seq
    data. Nat Commun 13, 3224 (2022). https://doi.org/10.1038/s41467-022-30755-0

Usage (from the QuantumXCT/python folder)::

    from LR_databases.lr_pairs import curate_lr_pairs
    lr = curate_lr_pairs(["PDGFB", "STAT3", "HES4", "TGFB1",
                          "PDGFRB", "TGFBR1", "TGFBR2"])
    # -> [('PDGFB', 'PDGFRB'), ('TGFB1', 'TGFBR1'), ('TGFB1', 'TGFBR2')]
"""
from __future__ import annotations

import os
import re
from itertools import product
from typing import List, Optional, Sequence, Tuple

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
MERGED_CSV = os.path.join(_HERE, "Human-merged-Jin-Dimitrov-LR-pairs.csv")


def _resolve(csv_path: Optional[str]) -> str:
    """An explicit path if it exists; else the same file name in this folder;
    else the merged database."""
    if csv_path:
        if os.path.exists(csv_path):
            return csv_path
        local = os.path.join(_HERE, os.path.basename(csv_path))
        if os.path.exists(local):
            return local
        raise FileNotFoundError(f"LR database not found: {csv_path}")
    return MERGED_CSV


def load_lr_pairs(csv_path: Optional[str] = None) -> pd.DataFrame:
    """Return the LR table as a DataFrame with columns ['ligand', 'receptor'].

    Entries may be complexes, with subunits joined by '_' (Dimitrov/LIANA),
    '&' (Jin/CellChat) or '+'.
    """
    df = pd.read_csv(_resolve(csv_path))
    if not {"ligand", "receptor"} <= set(df.columns):
        raise ValueError(f"expected 'ligand' and 'receptor' columns, found {list(df.columns)}")
    out = df[["ligand", "receptor"]].dropna().astype(str)
    return out.drop_duplicates().reset_index(drop=True)


def curate_lr_pairs(
    gene_list: Sequence[str],
    resource_csv: Optional[str] = None,
    complex_policy: str = "any",
    verbose: bool = True,
) -> List[Tuple[str, str]]:
    """Known LR pairs whose ligand and receptor are both among ``gene_list``.

    complex_policy:
      * "any" (default) keeps an interaction if at least one ligand subunit and
        one receptor subunit are in ``gene_list``, and returns edges only between
        those tested subunits (e.g. PDGFB -> PDGFRA_PDGFRB gives (PDGFB, PDGFRB)
        when only PDGFRB is tested).
      * "all" keeps it only if every subunit of both sides is tested.

    Returns a sorted, de-duplicated list of (ligand, receptor) gene tuples.
    """
    genes = {str(g).upper() for g in gene_list}
    try:
        table = load_lr_pairs(resource_csv)
    except (FileNotFoundError, ValueError) as e:
        if verbose:
            print(f"[curate_lr_pairs] {e} -> returning [] (no edges will be marked as LR).")
        return []

    pairs = set()
    for lig, rec in zip(table["ligand"], table["receptor"]):
        lig_tokens = [t.upper() for t in re.split(r"[_&+]", lig) if t]
        rec_tokens = [t.upper() for t in re.split(r"[_&+]", rec) if t]
        lig_in = [t for t in lig_tokens if t in genes]
        rec_in = [t for t in rec_tokens if t in genes]
        if complex_policy == "all":
            keep = len(lig_in) == len(lig_tokens) and len(rec_in) == len(rec_tokens)
        else:
            keep = bool(lig_in) and bool(rec_in)
        if keep:
            pairs.update((a, b) for a, b in product(lig_in, rec_in) if a != b)
    out = sorted(pairs)
    if verbose:
        print(f"[curate_lr_pairs] {len(out)} known LR pair(s) among the {len(genes)} genes: {out}")
    return out
