import numpy as np
import os
import json
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import StatePreparation

# Handle cases where __file__ is not defined (e.g., in interactive mode)
if '__file__' not in globals():
    __file__ = os.path.abspath('test4_evaluate_theta_stability.py')
from qiskit import transpile
from qiskit_aer import AerSimulator

# Test file saving at the beginning
script_dir = os.path.dirname(os.path.abspath(__file__))
test_file = os.path.join(script_dir, 'test_write_access.json')
try:
    with open(test_file, 'w') as f:
        json.dump({'test': 'write_access_test'}, f)
    os.remove(test_file)  # Clean up
    print("✓ Write access verified in directory:", script_dir)
except Exception as e:
    print("✗ Cannot write to directory:", script_dir)
    print("Error:", str(e))
    print("Please check directory permissions and try again.")
    exit(1)
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Target frequency vectors
pt_f_mo = [0.234577, 0.182927, 0.091105, 0.061693, 0.164634, 0.125897, 0.080703, 0.058465]
pt_c_mo = [0.118443, 0.006768, 0.048223, 0.002538, 0.078257, 0.005076, 0.053723, 0.006768, 0.140017, 0.013536, 0.096024, 0.013536, 0.170474, 0.017766, 0.195854, 0.032995]

# Target frequency vectors for optimization
pt_f_co = np.array([0.240181, 0.061934, 0.144260, 0.035498, 0.230363, 0.047583, 0.197130, 0.043051])
pt_c_co = np.array([0.115983, 0.019802, 0.053748, 0.019095, 0.077793, 0.022631, 0.056577, 0.025460, 
                    0.083451, 0.026167, 0.062235, 0.034653, 0.108911, 0.049505, 0.142150, 0.101839])

# Normalize target frequencies to ensure they sum to 1
pt_f_co = pt_f_co / np.sum(pt_f_co)
pt_c_co = pt_c_co / np.sum(pt_c_co)

frequencies = np.array(pt_f_mo)
normalized_probs = frequencies / np.sum(frequencies)
amplitudes1 = np.sqrt(normalized_probs)
qreg1 = QuantumRegister(3, 'q')
creg1 = ClassicalRegister(3, 'c')
qc1 = QuantumCircuit(qreg1, creg1)

state_prep = StatePreparation(amplitudes1)
qc1.append(state_prep, qreg1)
qc1.measure(qreg1, creg1)
print(f"\nQuantum Circuit (depth: {qc1.depth()}):")
print(qc1)


frequencies = np.array(pt_c_mo)
normalized_probs = frequencies / np.sum(frequencies)
amplitudes2 = np.sqrt(normalized_probs)
qreg2 = QuantumRegister(4, 'q')
creg2 = ClassicalRegister(4, 'c')
qc2 = QuantumCircuit(qreg2, creg2)

state_prep2 = StatePreparation(amplitudes2)
qc2.append(state_prep2, qreg2)
qc2.measure(qreg2, creg2)
print(f"\nQuantum Circuit (depth: {qc2.depth()}):")
print(qc2)

def create_entangled_circuit(qc1, qc2, thetas=None):
    """
    Create a circuit with CRY gates between two quantum circuits.
    
    Args:
        qc1: First quantum circuit (3 qubits)
        qc2: Second quantum circuit (4 qubits)
        thetas: List of 24 angles (in radians) for the CRY gates.
               First 12: from qc1 to qc2 (3*4=12)
               Next 12: from qc2 to qc1 (4*3=12, excluding self-loops)
               If None, uses pi/4 for all angles.
    
    Returns:
        QuantumCircuit: The combined circuit with CRY gates
    """
    # Merge the circuits
    merged_circuit = qc1 ^ qc2
    print(f"Merged circuit has {merged_circuit.num_qubits} qubits")  # Should be 7
    
    # Initialize thetas if not provided
    if thetas is None:
        thetas = [np.pi/4] * 24  # Default to pi/4 for all thetas
    elif len(thetas) != 24:
        raise ValueError("thetas must be a list of 24 values")
    
    # Add CRY gates from first 3 qubits to last 4 qubits
    theta_idx = 0
    for i in range(3):  # First 3 qubits (0,1,2)
        for j in range(3, 7):  # Last 4 qubits (3,4,5,6)
            merged_circuit.cry(thetas[theta_idx], i, j)
            theta_idx += 1
    
    # Add CRY gates from last 4 qubits to first 3 qubits
    for i in range(3, 7):  # Last 4 qubits (3,4,5,6)
        for j in range(3):  # First 3 qubits (0,1,2)
            if i != j:  # Avoid self-loops
                merged_circuit.cry(thetas[theta_idx], i, j)
                theta_idx += 1
    
    return merged_circuit, thetas

def print_theta_info(thetas):
    """Print information about the theta values used for CRY gates."""
    print("\nTheta values used for CRY gates:")
    print("From first 3 qubits to last 4 qubits (12 gates):")
    for i in range(12):
        print(f"  Theta {i+1:2d} (q{i//4}->q{i%4+3}): {thetas[i]:.4f}")
    
    print("\nFrom last 4 qubits to first 3 qubits (12 gates):")
    for i in range(12, 24):
        q_from = 3 + (i-12) // 3
        q_to = (i-12) % 3
        print(f"  Theta {i+1:2d} (q{q_from}->q{q_to}): {thetas[i]:.4f}")

def safe_log_ratio(p, q, epsilon=1e-10):
    """Safely compute log(p/q) with epsilon smoothing."""
    ratio = (p + epsilon) / (q + epsilon)
    return np.log(ratio + (ratio <= 0) * epsilon)

def jensen_shannon_divergence(p, q, epsilon=1e-10):
    """
    Calculate Jensen-Shannon Divergence between two probability distributions.
    More stable than KL divergence as it's symmetric and bounded.
    """
    p_safe = np.clip(p, epsilon, 1)
    q_safe = np.clip(q, epsilon, 1)
    
    # Normalize
    p_norm = p_safe / np.sum(p_safe)
    q_norm = q_safe / np.sum(q_safe)
    
    # Average distribution
    m = 0.5 * (p_norm + q_norm)
    
    # Calculate JS divergence
    js = 0.5 * (np.sum(p_norm * safe_log_ratio(p_norm, m)) + 
                np.sum(q_norm * safe_log_ratio(q_norm, m)))
    return js

def hellinger_distance(p, q, epsilon=1e-10):
    """
    Calculate Hellinger Distance between two probability distributions.
    Bounded between 0 and 1, more stable than KL divergence.
    """
    p_safe = np.clip(p, epsilon, 1)
    q_safe = np.clip(q, epsilon, 1)
    
    # Normalize
    p_norm = p_safe / np.sum(p_safe)
    q_norm = q_safe / np.sum(q_safe)
    
    # Calculate Hellinger distance
    return np.sqrt(0.5 * np.sum((np.sqrt(p_norm) - np.sqrt(q_norm)) ** 2))

def calculate_cost(thetas, qc1, qc2, shots=10000):
    """
    Calculate the cost using a combination of robust distance metrics.
    
    Args:
        thetas: List of 24 angles for CRY gates
        qc1: First quantum circuit (3 qubits)
        qc2: Second quantum circuit (4 qubits)
        shots: Number of shots for simulation
        
    Returns:
        float: Combined cost using multiple distance metrics
    """
    # Create circuit with current thetas
    circuit, _ = create_entangled_circuit(qc1, qc2, thetas)
    
    # Add measurements
    measured_circuit = circuit.copy()
    cr = ClassicalRegister(7, 'meas')
    measured_circuit.add_register(cr)
    measured_circuit.measure(range(7), range(7))
    
    # Simulate the circuit
    simulator = AerSimulator()
    compiled_circuit = transpile(measured_circuit, simulator)
    result = simulator.run(compiled_circuit, shots=shots).result()
    counts = result.get_counts()
    
    # Initialize frequency arrays
    measured_f3 = np.zeros(8)
    measured_c4 = np.zeros(16)
    total_shots = sum(counts.values())
    
    # Count frequencies
    for state, count in counts.items():
        state_rev = state[::-1]  # Reverse for Qiskit's little-endian
        first3 = int(state_rev[:3], 2)
        last4 = int(state_rev[3:7], 2)
        measured_f3[first3] += count / total_shots
        measured_c4[last4] += count / total_shots
    
    # Calculate multiple distance metrics
    js_f3 = jensen_shannon_divergence(pt_f_co, measured_f3)
    js_c4 = jensen_shannon_divergence(pt_c_co, measured_c4)
    
    hd_f3 = hellinger_distance(pt_f_co, measured_f3)
    hd_c4 = hellinger_distance(pt_c_co, measured_c4)
    
    # Combine metrics (JS divergence + Hellinger distance)
    # You can adjust these weights based on your needs
    total_cost = (js_f3 + js_c4) + 0.5 * (hd_f3 + hd_c4)
    
    # Add a small penalty for extreme theta values to improve stability
    theta_penalty = 0.01 * np.sum(np.sin(thetas) ** 2) / len(thetas)
    total_cost += theta_penalty
    
    return total_cost

def optimize_thetas(initial_thetas, qc1, qc2, maxiter=50):
    """
    Optimize thetas to match target frequencies, with theta values constrained between -π and π.
    
    Args:
        initial_thetas: Initial values for the 24 thetas (will be wrapped to [-π, π])
        qc1: First quantum circuit (3 qubits)
        qc2: Second quantum circuit (4 qubits)
        maxiter: Maximum number of iterations for optimization
        
    Returns:
        tuple: (optimized_thetas, result) where optimized_thetas is the list of optimized angles
               (wrapped to [-π, π]) and result is the full optimization result object
    """
    # Wrap initial thetas to [-π, π]
    initial_thetas = np.array(initial_thetas)
    initial_thetas = np.mod(initial_thetas + np.pi, 2*np.pi) - np.pi
    
    # Define constraint function to keep thetas in [-π, π]
    def theta_constraint(x):
        # This will be <= 0 when all thetas are in [-π, π]
        return np.pi - np.abs(x)
    
    # Create constraints for each theta
    constraints = [
        {'type': 'ineq', 'fun': lambda x, i=i: theta_constraint(x[i])}
        for i in range(len(initial_thetas))
    ]
    
    # Optimize using COBYLA method with constraints
    result = minimize(
        lambda x: calculate_cost(x, qc1, qc2),
        initial_thetas,
        method='COBYLA',
        constraints=constraints,
        options={'maxiter': maxiter, 'disp': True}
    )
    
    # Ensure the final thetas are in [-π, π]
    optimized_thetas = np.mod(result.x + np.pi, 2*np.pi) - np.pi
    
    return optimized_thetas, result

def run_multiple_optimizations(num_runs=100, maxiter=50, shots=10000):
    """
    Run the optimization multiple times and collect results.
    
    Args:
        num_runs: Number of optimization runs to perform
        maxiter: Maximum iterations per optimization
        shots: Number of shots for circuit simulation
        
    Returns:
        tuple: (all_thetas, all_costs, all_results)
            - all_thetas: numpy array of shape (num_runs, 24)
            - all_costs: numpy array of final costs for each run
            - all_results: list of optimization result objects
    """
    all_thetas = []
    all_costs = []
    all_results = []
    
    for run in range(1, num_runs + 1):
        print(f"\n{'='*80}")
        print(f"RUN {run}/{num_runs}")
        print(f"{'='*80}")
        
        # Use random initial thetas between -π and π for each run
        initial_thetas = np.random.uniform(-np.pi, np.pi, 24).tolist()
        
        # Run optimization
        optimized_thetas, result = optimize_thetas(initial_thetas, qc1, qc2, maxiter=maxiter)
        
        # Store results
        all_thetas.append(optimized_thetas)
        all_costs.append(result.fun)
        all_results.append(result)
        
        # Print progress
        print(f"Run {run:3d}/{num_runs}: Cost = {result.fun:.6f}", end="")
        print(f" | Success: {result.success}")
    
    return np.array(all_thetas), np.array(all_costs), all_results

def save_optimization_results(thetas, costs, results, output_file):
    """
    Save optimization results to a file.
    
    Args:
        thetas: numpy array of shape (num_runs, 24)
        costs: numpy array of final costs
        results: list of optimization result objects
        output_file: path to save the results
    """
    # Convert to lists for JSON serialization
    data = {
        'thetas': thetas.tolist(),
        'costs': costs.tolist(),
        'success': [bool(r.success) for r in results],
        'messages': [str(r.message) for r in results],
        'iterations': [int(r.nfev) for r in results],
        'timestamp': str(np.datetime64('now'))
    }
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Also save as numpy binary for easier analysis
    np.savez_compressed(
        output_file.replace('.json', '.npz'),
        thetas=thetas,
        costs=costs
    )
    
    return output_file

# Example usage with optimization
def run_single_optimization():
    print("\nStarting optimization to match target frequencies...")
    initial_thetas = [np.pi/4] * 24  # Initial guess
    optimized_thetas, result = optimize_thetas(initial_thetas, qc1, qc2)

    print("\nOptimization results:")
    print(f"Success: {result.success}")
    print(f"Final cost: {result.fun:.6f}")
    print(f"Number of iterations: {result.nfev}")

    # Create circuit with optimized thetas
    print("\nCreating circuit with optimized thetas:")
    optimized_circuit, _ = create_entangled_circuit(qc1, qc2, optimized_thetas)
    print_theta_info(optimized_thetas)

    # Create a copy of the circuit for measurement
    measured_circuit = optimized_circuit.copy()
    cr = ClassicalRegister(7, 'meas')
    measured_circuit.add_register(cr)
    measured_circuit.measure(range(7), range(7))

    # Simulate the circuit
    print("\nRunning simulation...")
    simulator = AerSimulator()
    compiled_circuit = transpile(measured_circuit, simulator)
    result = simulator.run(compiled_circuit, shots=10000).result()
    return result.get_counts()

# Main execution
if __name__ == "__main__":
    import datetime
    
    print("\n" + "="*80)
    print("QUANTUM CIRCUIT OPTIMIZATION")
    print("="*80)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    
    # Run multiple optimizations
    print("\n" + "="*80)
    print("RUNNING 10 OPTIMIZATION RUNS")
    print("="*80)
    
    all_thetas, all_costs, all_results = run_multiple_optimizations(
        num_runs=100,
        maxiter=50,
        shots=10000
    )
    
    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f'optimization_results_10runs_{timestamp}.json')
    
    try:
        saved_file = save_optimization_results(all_thetas, all_costs, all_results, output_file)
        print(f"\nOptimization results saved to: {saved_file}")
        print(f"Also saved as: {saved_file.replace('.json', '.npz')} for easier analysis")
        
        # Print summary statistics
        print("\n" + "="*80)
        print("OPTIMIZATION SUMMARY")
        print("="*80)
        print(f"Total runs: {len(all_costs)}")
        print(f"Successful runs: {sum(1 for r in all_results if r.success)}")
        print(f"Average final cost: {np.mean(all_costs):.6f} ± {np.std(all_costs):.6f}")
        print(f"Best cost: {np.min(all_costs):.6f}")
        print(f"Worst cost: {np.max(all_costs):.6f}")
        
        # Print best thetas
        best_idx = np.argmin(all_costs)
        print("\nBest thetas (lowest cost):")
        print_theta_info(all_thetas[best_idx])
        
        # Save best thetas separately
        best_thetas_file = os.path.join(output_dir, f'best_thetas_{timestamp}.json')
        with open(best_thetas_file, 'w') as f:
            json.dump({
                'thetas': all_thetas[best_idx].tolist(),
                'cost': float(all_costs[best_idx]),
                'run_index': int(best_idx),
                'success': bool(all_results[best_idx].success),
                'message': str(all_results[best_idx].message),
                'iterations': int(all_results[best_idx].nfev)
            }, f, indent=2)
        print(f"\nBest thetas saved to: {best_thetas_file}")
        
        # Run a single optimization with detailed output
        print("\n" + "="*80)
        print("RUNNING DETAILED SINGLE OPTIMIZATION")
        print("="*80)
        counts = run_single_optimization()
        
        # Process and display results
        first3_freq = {format(i, '03b'): 0 for i in range(8)}
        last4_freq = {format(i, '04b'): 0 for i in range(16)}
        total_shots = sum(counts.values())
        
        for state, count in counts.items():
            state_rev = state[::-1]  # Reverse for Qiskit's little-endian
            first3 = state_rev[:3]
            last4 = state_rev[3:7]
            first3_freq[first3] += count / total_shots
            last4_freq[last4] += count / total_shots
        
        print("\nMeasurement results:")
        print("First 3 qubits frequencies:", first3_freq)
        print("Last 4 qubits frequencies:", last4_freq)
        
    except Exception as e:
        print("\n" + "!"*60)
        print("ERROR: Failed to complete optimization:")
        print(str(e))
        print("Current working directory:", os.getcwd())