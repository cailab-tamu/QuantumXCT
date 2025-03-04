import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def create_joint_histogram(Xct1bool):
    """
    Creates a joint histogram of boolean columns from a NumPy array.
    """
    num_cols = Xct1bool.shape[1]
    joint_counts = Counter()
    bit_strings = set()

    for row in Xct1bool:
        bit_string = "".join(["1" if val else "0" for val in row])
        joint_counts[bit_string] += 1
        bit_strings.add(bit_string)

    return joint_counts, sorted(list(bit_strings))


def plot_joint_histogram(joint_counts, num_qubits, reverse_bits=False, features=None):
    """Plots the joint histogram, accounting for potential bit reversal."""
    
    all_bit_strings = [''.join(format(i, f'0{num_qubits}b')) for i in range(2**num_qubits)]
    
    counts = []
    mapped_bit_strings = []

    for bit_string in all_bit_strings:
        if reverse_bits:
            reversed_string = bit_string[::-1]  # Reverse bit order
            count_value = joint_counts.get(bit_string, 0)  # Extract count from original order
            mapped_bit_strings.append(reversed_string)  # Assign to reversed bit string
        else:
            count_value = joint_counts.get(bit_string, 0)
            mapped_bit_strings.append(bit_string)  # Keep regular order

        counts.append(count_value)

    # Sort bit strings by Hamming weight (count of '1's), then numerically
    sorted_pairs = sorted(zip(mapped_bit_strings, counts), key=lambda x: int(x[0], 2))

    # Unzip the sorted pairs
    sorted_bit_strings, sorted_counts = zip(*sorted_pairs)

    plt.figure(figsize=(10, 6))
    plt.bar(sorted_bit_strings, sorted_counts)

    # Create x-axis label with feature mapping
    if features is not None:
        if reverse_bits:
            xlabel_text = f"Bit String ("
            for i in range(num_qubits - 1, -1, -1):  # Iterate in reverse order for qn ... q0
                xlabel_text += f"q{i}={features[i]}, "
            xlabel_text = xlabel_text[:-2] + ")"  # Remove trailing comma and space
            plt.xlabel(xlabel_text)
        else:
            xlabel_text = f"Bit String ("
            for i in range(0, num_qubits):
                xlabel_text += f"q{i}={features[i]}, "
            xlabel_text = xlabel_text[:-2] + ")"
            plt.xlabel(xlabel_text)

    else:
        if reverse_bits:
            plt.xlabel(f"Bit String (q{num_qubits-1} ... q0)")
        else:
            plt.xlabel(f"Bit String (q0 ... q{num_qubits-1})")

        
    plt.ylabel("Frequency")
    plt.title("Joint Histogram of Boolean Columns")
    plt.xticks(rotation=45, ha="right")

    for i, count in enumerate(sorted_counts):
        plt.text(sorted_bit_strings[i], count, f"{count}", ha='center', va='bottom')

    plt.tight_layout()
    plt.show()