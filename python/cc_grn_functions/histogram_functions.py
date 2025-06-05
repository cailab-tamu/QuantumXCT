import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def create_joint_histogram(Xct1bool):
    """
    Creates a joint histogram of boolean columns from a NumPy array.
    """
    num_cols = Xct1bool.shape[1] # <--- Value is assigned here
    joint_counts = Counter()
    bit_strings = set()

    for row in Xct1bool:
        bit_string = "".join(["1" if val else "0" for val in row])
        joint_counts[bit_string] += 1
        bit_strings.add(bit_string)

    return joint_counts, sorted(list(bit_strings))


def create_percent_joint_histogram(Xct1bool):
    """
    Creates a joint histogram of boolean columns from a NumPy array,
    normalized by total number of counts and multiplied by 100 to get percentages.
    """
    num_rows, num_cols = Xct1bool.shape
    joint_counts = Counter()
    bit_strings = set()

    for row in Xct1bool:
        bit_string = "".join(["1" if val else "0" for val in row])
        joint_counts[bit_string] += 1
        bit_strings.add(bit_string)

    total_counts = num_rows #or sum(joint_counts.values()) if you want to be more robust.
    percent_joint_counts = {bit_string: (count / total_counts) * 100 for bit_string, count in joint_counts.items()}

    return percent_joint_counts, sorted(list(bit_strings))

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
    # Ensure it's a NumPy array and converted to boolean type
    boolean_vector = np.asarray(boolean_vector, dtype=bool)

    # Validate that the input is indeed 1-dimensional
    if boolean_vector.ndim != 1:
        raise ValueError(f"Input must be a 1-dimensional vector, but has {boolean_vector.ndim} dimensions.")

    # Convert boolean values to "0" or "1" strings before counting
    # This is done using a list comprehension or NumPy's vectorized operations
    bit_strings = np.where(boolean_vector, "1", "0")

    # Use collections.Counter to count the occurrences of "0" and "1"
    counts = Counter(bit_strings)

    return dict(counts) # Convert Counter to a regular dictionary for simpler output



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

    plt.figure(figsize=(12, 8))
    plt.bar(sorted_bit_strings, sorted_counts)

    # Create x-axis label with feature mapping
    if features is not None:
        if reverse_bits:
            xlabel_text = f"Bit String ("
            for i in range(num_qubits - 1, -1, -1):  # Iterate in reverse order for qn ... q0
                xlabel_text += f"q{i}={features[i]}, "
            xlabel_text = xlabel_text[:-2] + ")"  # Remove trailing comma and space
            plt.xlabel(xlabel_text, fontsize=14)
        else:
            xlabel_text = f"Bit String ("
            for i in range(0, num_qubits):
                xlabel_text += f"q{i}={features[i]}, "
            xlabel_text = xlabel_text[:-2] + ")"
            plt.xlabel(xlabel_text, fontsize=14)

    else:
        if reverse_bits:
            plt.xlabel(f"Bit String (q{num_qubits-1} ... q0)", fontsize=14)
        else:
            plt.xlabel(f"Bit String (q0 ... q{num_qubits-1})", fontsize=14)

        
    plt.title("Joint Histogram of Boolean Columns", fontsize=18)
    plt.xticks(rotation=45, ha="right", fontsize=14)
    plt.yticks(fontsize=14) # Increase y-ticks font size

    for i, count in enumerate(sorted_counts):
        if isinstance(count, float):
            plt.text(sorted_bit_strings[i], count, f"{count:.2f}", ha='center', va='bottom', fontsize=12)
            plt.ylabel("Frequency (%)", fontsize=16)

        else:
            plt.text(sorted_bit_strings[i], count, f"{count}", ha='center', va='bottom', fontsize=12)
            plt.ylabel("Frequency", fontsize=16)

    plt.tight_layout()
    plt.show()