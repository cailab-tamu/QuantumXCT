"""
qxct.encoding
=============
Quantum state encoding utilities for QuantumXCT.

Converts scRNA-seq boolean expression matrices into joint frequency histograms
that serve as quantum state amplitude inputs (Section 2.1 of the paper).

Functions
---------
create_joint_histogram          Raw joint frequency counts C(s)
create_percent_joint_histogram  Normalized joint counts as percentages
count_boolean_vector_occurrences  Marginal counts for a single gene
plot_joint_histogram            Bar-chart visualization of a histogram
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


def create_percent_joint_histogram(Xct1bool: np.ndarray) -> tuple[dict[str, float], list[str]]:
    """
    Creates a joint histogram of boolean columns from a NumPy array,
    normalized by total number of counts and multiplied by 100 to get percentages.
    Ensures bit strings are ordered MSB to LSB, where the LSB (rightmost bit)
    corresponds to the first feature/column (g0) in Xct1bool.

    Args:
        Xct1bool: A 2D NumPy array of boolean values (cells by genes).

    Returns:
        A tuple containing:
        - percent_joint_counts (dict[str, float]): A dictionary where keys are
          MSB-first bit strings and values are their percentages.
        - sorted_bit_strings (list[str]): A sorted list of unique bit strings encountered.
    """
    num_rows, num_cols = Xct1bool.shape
    joint_counts = Counter()
    unique_bit_strings = set()

    for row in Xct1bool:
        bit_string_list = []
        for val in reversed(row):
            bit_string_list.append("1" if val else "0")

        bit_string = "".join(bit_string_list)

        joint_counts[bit_string] += 1
        unique_bit_strings.add(bit_string)

    total_counts = sum(joint_counts.values())

    percent_joint_counts = {
        bit_string: (count / total_counts) * 100
        for bit_string, count in joint_counts.items()
    }

    sorted_bit_strings = sorted(list(unique_bit_strings))

    return percent_joint_counts, sorted_bit_strings


def count_boolean_vector_occurrences(boolean_vector):
    """
    Counts the occurrences of True and False values in a 1D boolean NumPy array,
    returning counts with "0" for False and "1" for True as keys.

    Args:
        boolean_vector (np.array): A 1-dimensional NumPy array of boolean values.

    Returns:
        dict: A dictionary where keys are string "0" (for False) or "1" (for True)
              and values are their counts.
    """
    boolean_vector = np.asarray(boolean_vector, dtype=bool)

    if boolean_vector.ndim != 1:
        raise ValueError(f"Input must be a 1-dimensional vector, but has {boolean_vector.ndim} dimensions.")

    bit_strings = np.where(boolean_vector, "1", "0")
    counts = Counter(bit_strings)

    return dict(counts)


def plot_joint_histogram(
    joint_counts: Counter,
    num_qubits: int,
    features: list = None,
    figsize=(5, 4),
    filename: str = None,
    title: str = 'Joint Histogram of Boolean Columns'
):
    """
    Plots the joint histogram, consistently in MSB-first order.

    Args:
        joint_counts: A Counter object of observed bit string counts (keys are MSB-first).
        num_qubits: The total number of qubits/features.
        features: Optional list of feature names, where features[i] maps to qubit q_i.
                  (i.e., features[0] for q0, features[1] for q1, etc.)
        figsize: Tuple specifying the figure size (width, height) in inches.
        filename: Optional string to save the plot to a file (e.g., 'my_histogram.png').
        title: String for the plot title.
    """
    all_bit_strings_msb_first = [''.join(format(i, f'0{num_qubits}b')) for i in range(2**num_qubits)]

    counts = [joint_counts.get(bs, 0) for bs in all_bit_strings_msb_first]
    sorted_bit_strings = all_bit_strings_msb_first
    sorted_counts = counts

    plt.figure(figsize=figsize)
    plt.bar(sorted_bit_strings, sorted_counts)

    xlabel_text = "Bit String ("
    if features is not None:
        feature_map_parts = []
        for i in range(num_qubits - 1, -1, -1):
            if i < len(features):
                feature_map_parts.append(f"q{i}={features[i]}")
            else:
                feature_map_parts.append(f"q{i}")
        xlabel_text += ", ".join(feature_map_parts) + ")"
    else:
        xlabel_text += f"q{num_qubits-1} ... q0)"

    plt.xlabel(xlabel_text)
    plt.title(title)
    plt.xticks(rotation=45, ha="right")

    for i, count in enumerate(sorted_counts):
        if isinstance(count, float):
            plt.text(sorted_bit_strings[i], count, f"{count:.2f}", ha='center', va='bottom')
            plt.ylabel("Frequency (%)")
        else:
            plt.text(sorted_bit_strings[i], count, f"{count}", ha='center', va='bottom')
            plt.ylabel("Frequency")

    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def normalize_counts(data_dict: dict) -> dict:
    """
    Normalizes a raw count dictionary to percentages (0–100 scale).

    Converts histogram counts (e.g. from create_joint_histogram) into
    percentage values suitable for side-by-side comparison of co-culture
    and mono-culture distributions.

    Args:
        data_dict (dict): Dictionary of {bitstring: count} or {key: numeric_value}.

    Returns:
        dict: Same keys, values rescaled so that they sum to 100.
              Returns all zeros if total count is 0.
    """
    total_count = sum(data_dict.values())
    if total_count == 0:
        return {key: 0 for key in data_dict}
    return {key: (value / total_count) * 100 for key, value in data_dict.items()}
