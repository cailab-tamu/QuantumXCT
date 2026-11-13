"""
qxct.scopes
===========
Biological-scope helpers for the CNOT/CRX topology search.

These are general-purpose utilities used by *any* optimization (the OO
orchestrator, the benchmark, or your own scripts) — they are not tied to a
particular backend. A candidate link (control, target) is either:

  - 'intra' — both qubits in the same cell register  → within-cell (GRN) link
  - 'inter' — qubits in different cell registers      → cross-cell (ligand-receptor / LR) link

Qubits [0, ng_circ1) belong to cell 1; qubits [ng_circ1, N) belong to cell 2.
"""

from __future__ import annotations

from typing import Dict, List, Tuple


def scope_of_cnot(control: int, target: int, ng_circ1: int) -> str:
    """Return 'intra' if both qubits are in the same cell register, else 'inter'."""
    c_cell = 0 if control < ng_circ1 else 1
    t_cell = 0 if target < ng_circ1 else 1
    return "intra" if c_cell == t_cell else "inter"


def filter_candidates_by_scope(
    candidates: List[Tuple[int, int]],
    ng_circ1: int,
    scope: str = "all",
) -> List[Tuple[int, int]]:
    """Filter candidate CNOTs by biological scope.

    scope:
        'all'   -> GRN + LR (no filtering).
        'intra' -> GRN only (within-cell links).
        'inter' -> LR only (cross-cell links).
    """
    if scope == "all":
        return list(candidates)
    if scope not in ("intra", "inter"):
        raise ValueError("scope must be 'all', 'intra' or 'inter'")
    return [(c, t) for (c, t) in candidates if scope_of_cnot(c, t, ng_circ1) == scope]


def enumerate_cnots(ng_total: int, ng_circ1: int, scope: str = "inter") -> List[Tuple[int, int]]:
    """Enumerate *all* directed CNOT pairs of a given scope, without any screening.

      - 'inter' -> every cross-register (ligand-receptor / LR) pair, both directions.
      - 'intra' -> every within-cell (GRN) pair.
      - 'all'   -> every distinct ordered pair.

    Used for the LR-only search, where the topology must consider all cross-cell
    links with no GRN links and no density-matrix pre-filter.
    """
    pairs: List[Tuple[int, int]] = []
    for c in range(ng_total):
        for t in range(ng_total):
            if c == t:
                continue
            if scope != "all" and scope_of_cnot(c, t, ng_circ1) != scope:
                continue
            pairs.append((c, t))
    return pairs


def split_candidates_by_cell(
    candidates: List[Tuple[int, int]],
    ng_circ1: int,
) -> Dict[str, List[Tuple[int, int]]]:
    """Split intra-cell (GRN) candidates into cell-1 and cell-2 groups.

    Useful for the per-cell sequential GRN optimization (optimize cell 1's
    internal topology, then cell 2's).
    """
    cell1, cell2 = [], []
    for (c, t) in candidates:
        if scope_of_cnot(c, t, ng_circ1) != "intra":
            continue
        if c < ng_circ1 and t < ng_circ1:
            cell1.append((c, t))
        else:
            cell2.append((c, t))
    return {"cell1": cell1, "cell2": cell2}
