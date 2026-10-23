"""
qxct.encoding
=============
Quantum state encoding utilities for QuantumXCT.

Converts scRNA-seq boolean expression matrices into joint frequency histograms
that serve as quantum state amplitude inputs (Section 2.1 of the paper).

Functions
---------
create_joint_histogram  Raw joint frequency counts C(s)
plot_joint_histogram    Bar-chart visualization of a histogram
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def create_joint_histogram(Xct1bool: np.ndarray):
    """
    Creates a joint histogram of boolean columns from a NumPy array.
    Ensures bit strings are ordered MSB to LSB, where the LSB (rightmost bit)
    corresponds to the first feature/column (g0) in Xct1bool.
    """
    num_cols = Xct1bool.shape[1]
    joint_counts = Counter()
    bit_strings = set()

    for row in Xct1bool:
        # Process the 'row' in reverse order to map g0 (first col) to LSB (rightmost char)
        # and gn (last col) to MSB (leftmost char).
        # This makes the string: "g_n_val ... g_1_val g_0_val"
        bit_string_list = []
        for val in reversed(row):  # Iterate from g_n to g_0
            bit_string_list.append("1" if val else "0")

        bit_string = "".join(bit_string_list)

        joint_counts[bit_string] += 1
        bit_strings.add(bit_string)

    return joint_counts, sorted(list(bit_strings))


def plot_joint_histogram(
    joint_counts: Counter,
    num_qubits: int,
    features: list = None,
    figsize=(5, 4),
    filename: str = None,
    title: str = 'Joint Histogram of Boolean Columns'
):
    """Plot the joint histogram (raw counts, MSB-first) via qxct.plotting.data_histogram
    so it matches every other figure. Signature unchanged."""
    import matplotlib.pyplot as plt
    from .plotting import data_histogram
    counts = {format(i, f'0{num_qubits}b'): joint_counts.get(format(i, f'0{num_qubits}b'), 0)
              for i in range(2 ** num_qubits)}
    fig, ax = plt.subplots(figsize=figsize)
    data_histogram(ax, counts, num_qubits, title)
    if features is not None:
        parts = [f"q{i}={features[i]}" if i < len(features) else f"q{i}" for i in range(num_qubits - 1, -1, -1)]
        ax.set_xlabel("Bit String (" + ", ".join(parts) + ")")
    fig.tight_layout()
    if filename:
        fig.savefig(filename, bbox_inches='tight', dpi=300)
    plt.show()
