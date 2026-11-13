"""
qxct.resources
==============
Ligand-receptor (LR) resource loading + curation.

:func:`curate_lr_pairs` intersects a curated LR database with the genes actually
present in a test and returns the surviving (ligand, receptor) pairs. This serves
two purposes:

  * plotting — deciding which cross-cell edges are drawn as validated LR links;
  * search  — restricting the QuantumXCT cross-cell candidate set to known LR
    pairs (the "LR-restricted" run), via :func:`restrict_inter_candidates_to_lr`.

The default database is the merged human Jin + Dimitrov resource shipped under
``LR_databases/Human-merged-Jin-Dimitrov-LR-pairs.csv`` (CellChat uses '&' for
complex subunits, LIANA/Dimitrov uses '_'; both are handled). If that file is
absent the older bundled consensus CSV is tried, then ``liana`` if importable.
"""
from __future__ import annotations

import os
from itertools import product
from typing import List, Optional, Sequence, Tuple

import pandas as pd

# preferred merged database, then the older bundled consensus CSV
_MERGED_CSV = "Human-merged-Jin-Dimitrov-LR-pairs.csv"
_DEFAULT_CSV = "lr_resource_dimitrov_consensus.csv"

# column-name aliases (lower-cased) for auto-detecting the two gene columns
_LIG_ALIASES = ("ligand", "ligand_symbol", "ligand_gene_symbol", "ligand_complex",
                "source_genesymbol", "source", "from", "lig", "ligand_gene")
_REC_ALIASES = ("receptor", "receptor_symbol", "receptor_gene_symbol", "receptor_complex",
                "target_genesymbol", "target", "to", "rec", "receptor_gene")


def _find_default_csv() -> Optional[str]:
    """Locate a bundled LR resource: the merged Jin+Dimitrov DB first (incl. a
    ``LR_databases/`` subfolder), then the older consensus CSV."""
    here = os.path.dirname(os.path.abspath(__file__))
    roots = [os.getcwd(), here, os.path.join(here, "..")]
    names = [os.path.join("LR_databases", _MERGED_CSV), _MERGED_CSV,
             os.path.join("LR_databases", _DEFAULT_CSV), _DEFAULT_CSV]
    for root in roots:
        for name in names:
            c = os.path.join(root, name)
            if os.path.exists(c):
                return c
    return None


def _pick_lr_columns(df: pd.DataFrame) -> Optional[Tuple[str, str]]:
    """Return (ligand_col, receptor_col) by matching known aliases, else None."""
    lower = {str(c).strip().lower(): c for c in df.columns}
    lig = next((lower[a] for a in _LIG_ALIASES if a in lower), None)
    rec = next((lower[a] for a in _REC_ALIASES if a in lower), None)
    if lig is not None and rec is not None:
        return lig, rec
    return None


def load_lr_resource(resource_csv: Optional[str] = None, resource: str = "consensus") -> pd.DataFrame:
    """Return a DataFrame with columns ['ligand', 'receptor'] (may contain complexes).

    Order of sources: an explicit ``resource_csv`` path, then a bundled DB
    (merged Jin+Dimitrov, else consensus), then liana if importable. Column names
    are auto-detected from common aliases. Raises FileNotFoundError if none found.
    """
    path = resource_csv or _find_default_csv()
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        if "resource" in df.columns and resource is not None and resource in set(df["resource"].astype(str)):
            df = df[df["resource"].astype(str) == resource]
        cols = _pick_lr_columns(df)
        if cols is None:
            raise ValueError(
                f"{path} lacks recognizable ligand/receptor columns "
                f"(have: {list(df.columns)}). Rename them to 'ligand'/'receptor' "
                f"or pass a file with known aliases.")
        out = df[[cols[0], cols[1]]].copy()
        out.columns = ["ligand", "receptor"]
        return out.dropna().drop_duplicates().reset_index(drop=True)

    # fallback: liana (only if the user happens to have it in this env)
    try:
        import liana
        df = liana.resource.select_resource(resource)
        df = df.rename(columns={"ligand_complex": "ligand", "receptor_complex": "receptor"})
        cols = _pick_lr_columns(df)
        if cols is None:
            raise ValueError("liana resource lacks recognizable ligand/receptor columns.")
        out = df[[cols[0], cols[1]]].copy(); out.columns = ["ligand", "receptor"]
        return out.dropna().drop_duplicates().reset_index(drop=True)
    except Exception as e:
        raise FileNotFoundError(
            f"No LR resource found. Place '{_MERGED_CSV}' (or '{_DEFAULT_CSV}') under "
            f"LR_databases/ or the working directory, or pass resource_csv=... "
            f"(liana fallback unavailable: {e})")


def curate_lr_pairs(
    gene_list: Sequence[str],
    resource_csv: Optional[str] = None,
    resource: str = "consensus",
    complex_policy: str = "any",
    verbose: bool = True,
) -> List[Tuple[str, str]]:
    """Curate LR pairs from a database, keeping only pairs drawable between genes
    that are within ``gene_list`` (the genes in your test).

    Receptors/ligands may be complexes (subunits joined by '_'). ``complex_policy``:
      * ``"any"`` (default) — keep the interaction if at least one ligand subunit
        AND one receptor subunit are tested, and emit edges only between the
        tested subunits. This is the plotting-friendly choice: e.g. the consensus
        entry PDGFB -> PDGFRA_PDGFRB yields (PDGFB, PDGFRB) when only PDGFRB is
        tested, matching a hand-curated PDGFB->PDGFRB.
      * ``"all"`` — stricter: keep only if every subunit of both is tested.

    Returns a sorted, de-duplicated list of (ligand, receptor) single-gene tuples.
    Intended for plotting only (which cross-cell edges are drawn as validated LR).
    """
    genes = {str(g).upper() for g in gene_list}
    try:
        res = load_lr_resource(resource_csv, resource)
    except FileNotFoundError as e:
        if verbose:
            print(f"[curate_lr_pairs] {e}\n  -> returning [] (network edges will be skipped).")
        return []

    import re as _re
    pairs = set()
    for lig, rec in zip(res["ligand"].astype(str), res["receptor"].astype(str)):
        # split complexes on any of _ & + so Jin ('&') and Dimitrov ('_') both work
        lig_tokens = [t.upper() for t in _re.split(r"[_&+]", lig) if t]
        rec_tokens = [t.upper() for t in _re.split(r"[_&+]", rec) if t]
        if not lig_tokens or not rec_tokens:
            continue
        lig_in = [t for t in lig_tokens if t in genes]
        rec_in = [t for t in rec_tokens if t in genes]
        if complex_policy == "all":
            keep = (len(lig_in) == len(lig_tokens)) and (len(rec_in) == len(rec_tokens))
        else:  # "any": at least one tested subunit on each side
            keep = bool(lig_in) and bool(rec_in)
        if keep:
            for a, b in product(lig_in, rec_in):
                if a != b:
                    pairs.add((a, b))
    out = sorted(pairs)
    if verbose:
        print(f"[curate_lr_pairs] {len(out)} LR pair(s) within the {len(genes)} test genes "
              f"(resource='{resource}'): {out}")
    return out
