"""
qxct.metrics
============
Edge- and direction-recovery metrics: how well a discovered topology matches a
known interaction network that YOU supply.

The discovered topology is the qubit-index edge list ``result.topology`` (a list
of ``(control, target)`` tuples). It is mapped to gene-level directed edges the
same way the relay-network plot does::

    raw_edges = [(gene_list[c], gene_list[t]) for c, t in topology]

so predicted edges are directly comparable to the ground-truth edge set.

Ground truth is always an explicit input (``ground_truth_edges``). For the
synthetic benchmark use ``simCellsSpace.ground_truth.synthetic_ground_truth()``;
for a real dataset with known interactions pass your own list of
``(source_gene, target_gene)`` tuples.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _auc_from_scores(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """(ROC-AUC, PR-AUC/average-precision). Uses sklearn if available, else a
    manual computation. Returns (nan, nan) if only one class is present."""
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan"), float("nan")
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))
    except Exception:
        # manual: sort by score desc, sweep thresholds
        order = np.argsort(-scores)
        y = labels[order]
        P, N = y.sum(), len(y) - y.sum()
        tp = np.cumsum(y); fp = np.cumsum(1 - y)
        tpr = tp / P; fpr = fp / N
        roc = float(np.trapz(np.concatenate([[0], tpr]), np.concatenate([[0], fpr])))
        prec = tp / (tp + fp); rec = tpr
        pr = float(np.trapz(np.concatenate([[prec[0]], prec]), np.concatenate([[0], rec])))
        return roc, pr


def _edge_scores_from_contribution(pred_edges, contribution_df, contribution_col):
    """Map each predicted directed gene edge to a contribution score.

    Parses the 'Gate' column ("LIG -> REC") of the isolated-contribution table.
    Default column preference: 'Marginal Contribution', then 'Isolated Reduction'.
    Higher score = more important. Edges absent from the table get the min score.
    """
    scores = {e: 0.0 for e in pred_edges}
    if contribution_df is None or "Gate" not in getattr(contribution_df, "columns", []):
        return scores, None
    col = contribution_col
    if col is None:
        for c in ("Marginal Contribution", "Isolated Reduction", "Isolated KL"):
            if c in contribution_df.columns:
                col = c
                break
    if col is None:
        return scores, None
    invert = col == "Isolated KL"  # for a raw-KL column, lower is better
    lut = {}
    for _, r in contribution_df.iterrows():
        gate = str(r["Gate"])
        if " -> " in gate:
            lig, rec = [s.strip() for s in gate.split(" -> ", 1)]
            v = float(r[col])
            lut[(lig, rec)] = -v if invert else v
    if lut:
        lo = min(lut.values())
        scores = {e: lut.get(e, lo) for e in pred_edges}
    return scores, col


def evaluate_edge_recovery(
    topology: List[Tuple[int, int]],
    gene_list: Sequence[str],
    ground_truth_edges: Optional[List[Tuple[str, str]]] = None,
    contribution_df: Optional[pd.DataFrame] = None,
    contribution_col: Optional[str] = None,
    out_dir: Optional[str] = None,
    scope_tag: str = "edge",
    show: bool = True,
) -> Dict:
    """Evaluate how well ``topology`` recovers the ground-truth network.

    ``ground_truth_edges`` (required): the known directed edges, as gene-name
    tuples ``[(source, target), ...]``.

    Returns a dict with precision, recall (directed), direction accuracy,
    ROC-AUC and PR-AUC, plus per-edge and summary DataFrames. If ``out_dir`` is
    given, saves the tables (CSV) and a PR/ROC figure there.
    """
    if not ground_truth_edges:
        raise ValueError(
            "evaluate_edge_recovery needs ground_truth_edges=[(source_gene, target_gene), ...]. "
            "For the synthetic benchmark use "
            "simCellsSpace.ground_truth.synthetic_ground_truth().")

    # 1) qubit-index topology -> gene-level directed edges (same mapping as the plot)
    pred_edges = [(gene_list[c], gene_list[t]) for c, t in topology]
    pred_set = list(dict.fromkeys(pred_edges))          # dedupe, keep order
    gt = list(dict.fromkeys(ground_truth_edges))
    gt_set = set(gt)
    gt_undir = {frozenset(e) for e in gt}
    genes = set(map(str, gene_list))
    gt_testable = [e for e in gt if e[0] in genes and e[1] in genes]

    # 2) directed precision / recall
    tp_dir = [e for e in pred_set if e in gt_set]
    precision = len(tp_dir) / len(pred_set) if pred_set else float("nan")
    recall = len(tp_dir) / len(gt) if gt else float("nan")
    recall_testable = (len([e for e in tp_dir if e in set(gt_testable)]) / len(gt_testable)
                       if gt_testable else float("nan"))

    # 3) direction accuracy: among predicted edges whose UNORDERED pair is a GT
    #    edge, the fraction with the correct orientation.
    recovered_undir = [e for e in pred_set if frozenset(e) in gt_undir]
    direction_accuracy = (len(tp_dir) / len(recovered_undir)) if recovered_undir else float("nan")

    # 4) AUC: rank predicted edges by contribution, label by GT membership
    scores_map, used_col = _edge_scores_from_contribution(pred_set, contribution_df, contribution_col)
    labels = np.array([1 if e in gt_set else 0 for e in pred_set])
    scores = np.array([scores_map[e] for e in pred_set], dtype=float)
    roc_auc, pr_auc = _auc_from_scores(labels, scores)

    # hypergeometric null: TP expected if |pred| directed pairs were drawn at
    # random from the n(n-1) ordered pairs of the panel
    from scipy.stats import hypergeom
    n_pairs = len(genes) * (len(genes) - 1)
    gt_t = set(gt_testable)
    tp_t = len([e for e in pred_set if e in gt_t])
    null_expected_tp = len(pred_set) * len(gt_t) / n_pairs if n_pairs else float("nan")
    hypergeom_p = float(hypergeom.sf(tp_t - 1, n_pairs, len(gt_t), len(pred_set))) if n_pairs else float("nan")

    per_edge = pd.DataFrame({
        "predicted_edge": [f"{a}->{b}" for a, b in pred_set],
        "in_ground_truth": labels.astype(int),
        "undirected_match": [1 if frozenset(e) in gt_undir else 0 for e in pred_set],
        "contribution": scores,
    })
    summary = pd.DataFrame([{
        "scope": scope_tag,
        "n_predicted": len(pred_set),
        "n_ground_truth": len(gt),
        "n_ground_truth_testable": len(gt_testable),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "recall_testable": round(recall_testable, 4) if recall_testable == recall_testable else np.nan,
        "direction_accuracy": round(direction_accuracy, 4) if direction_accuracy == direction_accuracy else np.nan,
        "roc_auc": round(roc_auc, 4) if roc_auc == roc_auc else np.nan,
        "pr_auc": round(pr_auc, 4) if pr_auc == pr_auc else np.nan,
        "null_expected_TP": round(null_expected_tp, 3),
        "hypergeom_p": round(hypergeom_p, 4),
        "contribution_col": used_col,
    }])

    result = {"summary": summary, "per_edge": per_edge,
              "precision": precision, "recall": recall, "recall_testable": recall_testable,
              "direction_accuracy": direction_accuracy, "roc_auc": roc_auc, "pr_auc": pr_auc,
              "ground_truth": gt, "predicted": pred_set,
              "true_positives": tp_dir}

    if out_dir:
        import os
        os.makedirs(out_dir, exist_ok=True)
        summary.to_csv(os.path.join(out_dir, f"{scope_tag}_recovery_summary.csv"), index=False)
        per_edge.to_csv(os.path.join(out_dir, f"{scope_tag}_recovery_per_edge.csv"), index=False)
        _plot_recovery(result, out_dir, scope_tag, show)
    return result


def _plot_recovery(result, out_dir, scope_tag, show):
    import os
    import matplotlib.pyplot as plt

    pe = result["per_edge"].sort_values("contribution", ascending=False).reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # (a) predicted edges ranked by contribution, colored by GT membership
    colors = ["#56B4E9" if v else "#E69F00" for v in pe["in_ground_truth"]]  # blue=in GT, orange=FP (colorblind-safe)
    axes[0].bar(range(len(pe)), pe["contribution"], color=colors, alpha=0.85)
    axes[0].set_xticks(range(len(pe)))
    axes[0].set_xticklabels(pe["predicted_edge"], rotation=90, fontsize=8)
    axes[0].set_ylabel("contribution (KL)")
    axes[0].set_title(f"[{scope_tag}] predicted edges (blue=in ground truth, orange=false positive)")

    # (b) metric bars
    s = result["summary"].iloc[0]
    names = ["precision", "recall", "direction_accuracy", "roc_auc", "pr_auc"]
    vals = [s[n] if s[n] == s[n] else 0.0 for n in names]
    axes[1].bar(names, vals, color="#56B4E9", alpha=0.85)
    axes[1].set_ylim(0, 1.05)
    for i, v in enumerate(vals):
        axes[1].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    axes[1].set_title(f"[{scope_tag}] recovery metrics")
    axes[1].tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{scope_tag}_recovery_metrics.png"), dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)
