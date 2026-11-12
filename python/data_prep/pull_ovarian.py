"""Rebuild the annotated ovarian h5ad from the public raw counts (reproduction step).

This is a one-time DATA-CONSTRUCTION step, run by hand BEFORE the benchmark. It is
deliberately NOT part of run_all.py.

What it does
------------
1. Downloads the raw single-cell counts for GEO series GSE224333 (in-vitro
   monoculture + co-culture scRNA-seq of ovarian clear-cell carcinoma cells and
   fibroblasts).
2. Re-attaches the preserved cell annotations that ship with this repo
   (``dataset_co_mo2_cell_metadata.csv`` + ``dataset_co_mo2_genes.txt``, exported
   once from the authors' annotated object by ``export_metadata.py``), joining on
   the cell barcode WITHIN each sample -> BatchID group.
3. Writes ``dataset_co_mo2.h5ad`` with raw integer counts in ``X``, the preserved
   ``obs`` (BatchID / CellType / ClusterID / CellCycle) and the preserved gene order
   -- exactly the shape ``analysis_tools/data.load_ovarian`` expects.

Why a per-group join: the integrated object uses plain 10x barcodes (e.g.
``AAACCCACAGTGCCTG-1``) with no per-sample suffix, so the same barcode can occur in
more than one GEO sample. BatchID says which sample each annotated cell came from,
so each group is joined only against its own sample's matrix.

Run on your machine (where GEO is reachable), from QuantumXCT/python:

    # 1) inspect the series first (no download): confirm the GSM -> BatchID mapping
    python data_prep/pull_ovarian.py --dry-run

    # 2) build it
    python data_prep/pull_ovarian.py --out dataset_co_mo2.h5ad

If the auto-mapping from GSM titles is wrong, pass an explicit map, e.g.
    python data_prep/pull_ovarian.py --map GSM1234=Cancer Cells (Mo) GSM1235=...

Requires: GEOparse, scanpy, anndata, pandas, scipy (pip install GEOparse scanpy).
"""
from __future__ import annotations
import argparse
import gzip
import os
import re
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

GSE = "GSE224333"
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_META = os.path.join(HERE, "dataset_co_mo2_cell_metadata.csv")
DEFAULT_GENES = os.path.join(HERE, "dataset_co_mo2_genes.txt")

# the four integrated groups, as BatchID values in the preserved metadata
BATCHES = ["Cancer Cells (Mo)", "Cancer Cells (Co)", "Fibroblasts (Mo)", "Fibroblasts (Co)"]


def classify_title(title: str):
    """Infer a BatchID from a GSM sample title by keyword. Returns None if unsure."""
    t = title.lower()
    if any(k in t for k in ("fibroblast", "caf", "stroma")):
        lineage = "Fibroblasts"
    elif any(k in t for k in ("cancer", "tumor", "tumour", "carcinoma", "occc", "epithel")):
        lineage = "Cancer Cells"
    else:
        return None
    # co-culture vs monoculture (check 'co' carefully: match word, not substring)
    if re.search(r"\b(co[-\s]?cult|coculture|co\b|together|mixed)", t):
        state = "Co"
    elif re.search(r"\b(mono|monocult|alone|single|solo)\b", t):
        state = "Mo"
    else:
        return None
    return f"{lineage} ({state})"


def _read_10x_group(files, tmp):
    """Stage a GSM's 10x triplet into standard names and read it with scanpy."""
    import scanpy as sc
    picks = {"matrix": None, "barcodes": None, "features": None}
    for f in files:
        low = os.path.basename(f).lower()
        if low.endswith(".mtx") or low.endswith(".mtx.gz"):
            picks["matrix"] = f
        elif "barcode" in low:
            picks["barcodes"] = f
        elif "feature" in low or "gene" in low:
            picks["features"] = f
    if not all(picks.values()):
        raise RuntimeError(f"Could not find a 10x triplet among: {[os.path.basename(x) for x in files]}")
    d = tempfile.mkdtemp(dir=tmp)
    for std, src in (("matrix.mtx.gz", picks["matrix"]),
                     ("barcodes.tsv.gz", picks["barcodes"]),
                     ("features.tsv.gz", picks["features"])):
        dst = os.path.join(d, std)
        if src.endswith(".gz"):
            shutil.copy(src, dst)
        else:  # gzip a plain file into the expected name
            with open(src, "rb") as fi, gzip.open(dst, "wb") as fo:
                shutil.copyfileobj(fi, fo)
    a = sc.read_10x_mtx(d, var_names="gene_symbols", make_unique=True)
    a.var_names_make_unique()
    return a


def main():
    ap = argparse.ArgumentParser(description="Rebuild dataset_co_mo2.h5ad from GSE224333 + preserved metadata.")
    ap.add_argument("--out", default=os.path.join(os.getcwd(), "dataset_co_mo2.h5ad"))
    ap.add_argument("--meta", default=DEFAULT_META, help="preserved cell metadata CSV")
    ap.add_argument("--genes", default=DEFAULT_GENES, help="preserved gene-order txt")
    ap.add_argument("--workdir", default=os.path.join(tempfile.gettempdir(), "gse224333"),
                    help="where GEOparse downloads the series")
    ap.add_argument("--dry-run", action="store_true", help="list GSMs, titles and files; do not download matrices or build")
    ap.add_argument("--map", nargs="*", default=[], metavar="GSM=BatchID",
                    help='explicit GSM -> BatchID overrides, e.g. GSM123="Cancer Cells (Mo)"')
    args = ap.parse_args()

    try:
        import GEOparse
    except ImportError:
        sys.exit("GEOparse not installed. Run: pip install GEOparse")

    os.makedirs(args.workdir, exist_ok=True)
    print(f"[pull_ovarian] fetching {GSE} metadata via GEOparse ...", flush=True)
    gse = GEOparse.get_GEO(geo=GSE, destdir=args.workdir, silent=True)

    overrides = {}
    for kv in args.map:
        k, v = kv.split("=", 1)
        overrides[k.strip()] = v.strip()

    # map every GSM to a BatchID (override first, else infer from title)
    gsm_batch, listing = {}, []
    for gsm_id, gsm in gse.gsms.items():
        title = gsm.metadata.get("title", [""])[0]
        supp = [u for u in gsm.metadata.get("supplementary_file", []) + gsm.metadata.get("supplementary_file_1", [])]
        batch = overrides.get(gsm_id) or classify_title(title)
        gsm_batch[gsm_id] = batch
        listing.append((gsm_id, title, batch, supp))

    print("\n[pull_ovarian] samples in the series:")
    for gsm_id, title, batch, supp in listing:
        print(f"  {gsm_id}  ->  {batch!r:22}  | {title}")
        for u in supp:
            print(f"        {os.path.basename(u)}")

    assigned = {b: [g for g, bb in gsm_batch.items() if bb == b] for b in BATCHES}
    print("\n[pull_ovarian] BatchID assignment:")
    for b in BATCHES:
        print(f"  {b:22} <- {assigned[b] or 'NONE (fix with --map)'}")
    unknown = [g for g, b in gsm_batch.items() if b is None]
    if unknown:
        print(f"\n[pull_ovarian] WARNING: unmapped GSMs {unknown}. Use --map to assign them.")
    if args.dry_run:
        print("\n[pull_ovarian] dry run only; no matrices downloaded, no h5ad written.")
        return
    if any(not assigned[b] for b in BATCHES) or unknown:
        sys.exit("Refusing to build: every BatchID must map to exactly one GSM. Re-run with --map.")

    # preserved annotations + gene order
    meta = pd.read_csv(args.meta)
    bc_col = "CellID" if "CellID" in meta.columns else meta.columns[0]
    meta[bc_col] = meta[bc_col].astype(str)
    meta = meta.set_index(bc_col, drop=False)
    genes = [g.strip() for g in open(args.genes) if g.strip()]
    print(f"\n[pull_ovarian] preserved metadata: {len(meta)} cells, {len(genes)} genes; "
          f"BatchIDs: {dict(meta['BatchID'].value_counts())}")

    import anndata as ad
    import scanpy as sc
    from scipy.sparse import vstack, csr_matrix

    gse.download_supplementary_files(directory=args.workdir, download_sra=False)

    # read each sample's matrix, subset to that group's barcodes, assemble in metadata order
    blocks, obs_order = [], []
    for b in BATCHES:
        gsm_id = assigned[b][0]
        files = []
        for root, _d, fs in os.walk(args.workdir):
            for f in fs:
                if gsm_id in f:
                    files.append(os.path.join(root, f))
        a = _read_10x_group(files, args.workdir)
        a.obs_names = [str(x) for x in a.obs_names]
        want = meta.index[meta["BatchID"] == b].tolist()
        have = a.obs_names.intersection(pd.Index(want))
        missing = sorted(set(want) - set(have))
        print(f"  {b:22}: sample {gsm_id}, {a.n_obs} cells in GEO, "
              f"{len(have)}/{len(want)} annotated barcodes matched"
              + (f"  ({len(missing)} missing!)" if missing else ""))
        if missing:
            print(f"      first missing: {missing[:5]}")
        sub = a[pd.Index(want)]                      # metadata cell order; raises if any barcode absent
        # reindex genes to the preserved order in one block assignment (absent symbols -> zero column)
        import scipy.sparse as sp
        M = sub.X.tocsc()
        var_pos = {g: i for i, g in enumerate(sub.var_names)}
        src = [var_pos.get(g) for g in genes]
        dst = [j for j, c in enumerate(src) if c is not None]
        keep = [c for c in src if c is not None]
        Xg = sp.lil_matrix((sub.n_obs, len(genes)), dtype=np.float32)
        if keep:
            Xg[:, dst] = M[:, keep]
        blocks.append(Xg.tocsr())
        obs_order.extend(want)
        n_missing_genes = len(genes) - len(dst)
        if n_missing_genes:
            print(f"      {n_missing_genes} preserved genes absent in this sample -> zero-filled")

    X = vstack(blocks).tocsr()
    obs = meta.loc[obs_order].copy()
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes))
    for c in ("BatchID", "CellType", "CellCycle"):
        if c in adata.obs:
            adata.obs[c] = adata.obs[c].astype("category")

    assert adata.n_obs == len(meta), f"expected {len(meta)} cells, got {adata.n_obs}"
    assert adata.n_vars == len(genes), f"expected {len(genes)} genes, got {adata.n_vars}"
    adata.write_h5ad(args.out)
    print(f"\n[pull_ovarian] wrote {args.out}: {adata.shape[0]} cells x {adata.shape[1]} genes")
    print("[pull_ovarian] sanity: run  python -c \"from analysis_tools.data import load_ovarian; "
          "load_ovarian(); print('load_ovarian OK')\"")


if __name__ == "__main__":
    main()
