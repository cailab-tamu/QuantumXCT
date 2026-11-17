"""
simCellsSpace.ground_truth
==========================
The interaction network programmed into the synthetic tissue simulator.

This is the *answer key* for the synthetic benchmark only. It lives next to the
simulator (not inside the ``qxct`` package) because it describes this specific
dataset, not QuantumXCT. To score a QuantumXCT topology against it::

    from qxct.metrics import evaluate_edge_recovery
    from simCellsSpace.ground_truth import synthetic_ground_truth

    rec = evaluate_edge_recovery(result.topology, gene_list,
                                 ground_truth_edges=synthetic_ground_truth())

For a real dataset with known interactions, pass your own
``ground_truth_edges=[("LIGAND", "RECEPTOR"), ...]`` instead.
"""
from __future__ import annotations

from typing import List, Tuple

# Directed edges programmed in the simulator (simCellsSpace/simulate_cells.ipynb). 70->71, 70->72 and 73->76 complete the
# testable ground truth for the 6-gene panel {50, 90 | 60, 70, 71, 80}.
SIMULATOR_EDGES = [(50, 60),                       # forward LR       (cell type A -> B)
                   (60, 70),                       # receptor -> master regulator (B)
                   (70, 71), (70, 72), (70, 80),   # reactive GRN     (B)
                   (80, 90),                       # retrograde LR    (B -> A)
                   (73, 74), (73, 75), (73, 76)]   # autonomous module (no input)


def synthetic_ground_truth(gene_prefix: str = "Gene_",
                           include_feedback_closure: bool = False,
                           transitive: bool = False) -> List[Tuple[str, str]]:
    """Directed ground-truth edges of the synthetic benchmark, as gene names.

    transitive=False : only the direct edges programmed in the simulator.
    transitive=True  : every ordered pair (a, b) joined by a directed causal path
                       a ~> b (an upstream-downstream CRX is a valid "shortcut").
    include_feedback_closure : also add 90->50. This is NOT a simulator edge;
                       use it only as a sensitivity analysis.
    """
    edges = list(SIMULATOR_EDGES)
    if include_feedback_closure:
        edges.append((90, 50))
    if transitive:
        import networkx as nx
        G = nx.DiGraph(edges)
        edges = [(a, b) for a in G for b in G if a != b and nx.has_path(G, a, b)]
    return [(f"{gene_prefix}{a}", f"{gene_prefix}{b}") for a, b in edges]
