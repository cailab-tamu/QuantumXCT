import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.visualization import plot_histogram

def create_grn_ansatz(ng, cell_type):
    """
    Creates a parameterized quantum circuit representing a Gene Regulatory Network (GRN) ansatz.

    Args:
        ng: The number of genes in the network.
        cell_type: A string identifier for the cell type (e.g., "CT1", "CT2").

    Returns:
        A QuantumCircuit object representing the GRN ansatz.
    """
    cell_type = cell_type.lower()
    ansatz_grn = QuantumCircuit(ng, name=f"{cell_type}_GRN_Ansatz")

    # Gene activation probabilities (RY rotations after Hadamard)
    params_act = [Parameter(f'{cell_type}_act_{i}') for i in range(ng)]
    params_post_act = [Parameter(f'{cell_type}_post_acti_{i}') for i in range(ng)]
    params_post_act2 = [Parameter(f'{cell_type}_post_acti2_{i}') for i in range(ng)]
    for i in range(ng):
        ansatz_grn.h(i)
        ansatz_grn.ry(params_post_act[i], i)
        #ansatz_grn.rx(params_post_act[i], i)
        #ansatz_grn.rz(params_post_act[i], i)
        ansatz_grn.rz(params_act[i], i)  # Use RZ for activation
        #ansatz_grn.ry(params_post_act2[i], i)
        ansatz_grn.rx(params_post_act2[i], i)
        #ansatz_grn.rz(params_post_act2[i], i)

    # Gene interaction CRX gates
    for i in range(ng):
        for j in range(ng):
            if i != j:
                param_name = f'{cell_type}_grn_{i}_{j}'
                param = Parameter(param_name)
                ansatz_grn.cry(param, i, j)
                #ansatz_grn.crx(param, i, j)

    return ansatz_grn

def create_circuit_lr2(ansatz_grn_ct1, ansatz_grn_ct2, cell_type1='ct1', cell_type2='ct2', interactions=None):
    """Concatenates two circuits and includes interactions using CRY rotations with parameterized angles."""
    ng_ct1 = ansatz_grn_ct1.num_qubits
    ng_ct2 = ansatz_grn_ct2.num_qubits
    cell_type1 = cell_type1.lower()
    cell_type2 = cell_type2.lower()
    num_features = ng_ct1 + ng_ct2
    ccgrn_circuit = QuantumCircuit(num_features, name=f"CC_GRN_Ansatz")

    # Gene activation probabilities (RZ rotations after Hadamard) for CT1
    params_act_ct1 = [Parameter(f'{cell_type1}_act_{i}') for i in range(ng_ct1)]
    params_post_act_ct1 = [Parameter(f'{cell_type1}_post_acti_{i}') for i in range(ng_ct1)]
    params_post_act2_ct1 = [Parameter(f'{cell_type1}_post_acti2_{i}') for i in range(ng_ct1)]
    for i in range(ng_ct1):
        #ccgrn_circuit.h(i)
        ccgrn_circuit.ry(params_post_act_ct1[i], i)
        #ccgrn_circuit.rx(params_post_act_ct1[i], i)
        ccgrn_circuit.rz(params_act_ct1[i], i)
        #ccgrn_circuit.ry(params_post_act2_ct1[i], i)
        #ccgrn_circuit.rx(params_post_act2_ct1[i], i)
        #ccgrn_circuit.rz(params_post_act2_ct1[i], i)

    # Gene interaction CRX gates for ct1
    for i in range(ng_ct1):
        for j in range(ng_ct1):
            if i != j:
                param_name = f'{cell_type1}_grn_{i}_{j}'
                param = Parameter(param_name)
                ccgrn_circuit.cry(param, i, j)
                #ccgrn_circuit.crx(param, i, j)

    # Gene activation probabilities (RZ rotations after Hadamard) for CT2
    params_act_ct2 = [Parameter(f'{cell_type2}_act_{i}') for i in range(ng_ct2)]
    params_post_act_ct2 = [Parameter(f'{cell_type2}_post_acti_{i}') for i in range(ng_ct2)]
    #params_post_act2_ct2 = [Parameter(f'{cell_type2}_post_acti2_{i}') for i in range(ng_ct2)]
    for i, j in enumerate(range(ng_ct1, num_features)):
        #ccgrn_circuit.h(j)
        ccgrn_circuit.ry(params_post_act_ct2[i], j)
        #ccgrn_circuit.rx(params_post_act_ct2[i], j)
        ccgrn_circuit.rz(params_act_ct2[i], j)  # Corrected indexing here
        #ccgrn_circuit.ry(params_post_act2_ct2[i], j)
        #ccgrn_circuit.rx(params_post_act2_ct2[i], j)
        #ccgrn_circuit.rz(params_post_act2_ct2[i], j)

    # Gene interaction CRX gates for ct2
    for i, q1 in enumerate(range(ng_ct1, num_features)):
        for j, q2 in enumerate(range(ng_ct1, num_features)):
            if q1 != q2:
                param_name = f'{cell_type2}_grn_{i}_{j}'
                param = Parameter(param_name)
                ccgrn_circuit.cry(param, q1, q2)
                #ccgrn_circuit.crx(param, q1, q2)

    # Add interactions if provided LR info here
    if interactions:
        for (q1, q2) in interactions.keys():
            if not (0 <= q1 < num_features and 0 <= q2 < num_features):
                raise ValueError("Qubit indices in interactions are out of range.")
            param_name = f"lr_{q1}_{q2}"
            angle_param = Parameter(param_name)
            ccgrn_circuit.crx(angle_param, q1, q2)
            #ccgrn_circuit.cry(angle_param, q1, q2)

    return ccgrn_circuit

# def create_interaction_observable_from_histogram(joint_counts: Counter, num_features: int, min_ones: int = 1, rm_all_ones: bool=False):
#     """Creates a SparsePauliOp from joint histogram counts,
#         favoring bit strings with odd number of '1's.

#     Args:
#         joint_counts: A Counter object from create_joint_histogram.
#         num_features: The total number of qubits.
#         min_ones: The minimum number of '1's required in a bit string
#                   for the interaction to be included.
#         rm_all_ones: If True, removes the bit string with all '1's.
#     Returns:
#         A SparsePauliOp observable.
#     """
#     interaction_strength_list = []

#     for bit_string, count in joint_counts.items():
#         num_ones = bit_string.count('1')  # Count the number of '1's

#         if num_ones == num_features and rm_all_ones:
#             continue

#         if num_ones >= min_ones:  # Consider only if at least min_ones '1's are present
#             nodes = tuple(i for i, bit in enumerate(bit_string) if bit == '1')

#             strength = -float(count)
#             #strength = count*(-1.0)**(num_ones+1) 
#             #strength = float(count)
#             #strength = count*(-1.0)**(num_ones) 

#             pauli_string = ""
#             for i in range(num_features):
#                 if i in nodes:
#                     pauli_string += "Z"
#                 else:
#                     pauli_string += "I"

#             interaction_strength_list.append((pauli_string, strength))

#     interaction_observable = SparsePauliOp.from_list(interaction_strength_list)
#     return interaction_observable



from qiskit.quantum_info import SparsePauliOp
from collections import Counter
import itertools # Used for generating all Pauli string combinations
import numpy as np # For numerical precision and calculations

# --- NEW HELPER FUNCTION (placed inside for self-containment, or can be external) ---
def _calculate_pauli_z_eigenvalue_for_basis_state(pauli_string_z_only: str, bit_string: str) -> int:
    """
    Calculates the eigenvalue of a Pauli string (composed ONLY of Z and I)
    for a given computational basis bit string.
    
    Assumes pauli_string_z_only and bit_string are both ordered from MSB to LSB
    (e.g., "ZIZ" where Z is for q2, I for q1, Z for q0, and "011" where 0 is for q2, 1 for q1, 1 for q0).
    """
    if len(pauli_string_z_only) != len(bit_string):
        raise ValueError("Pauli string length must match bit string length.")

    eigenvalue = 1
    # Iterate from MSB (index 0) to LSB (index num_features-1)
    for i in range(len(pauli_string_z_only)):
        pauli_op = pauli_string_z_only[i].upper() # Pauli op at this qubit position
        bit_val = int(bit_string[i]) # Bit value at this qubit position

        if pauli_op == 'Z':
            eigenvalue *= (1 - 2 * bit_val) # (+1 if bit is 0, -1 if bit is 1)
        # For 'I', eigenvalue *= 1, so no change needed
            
    return eigenvalue


# --- MODIFIED create_interaction_observable_from_histogram ---
def create_interaction_observable_from_histogram(
    joint_counts: Counter,
    num_features: int,
    min_ones: int = 0, # Kept as per your original signature
    # Added new parameters for fine-grained control over energy assignment
    unobserved_punishment: float = 10.0,
    normalization_offset: float = 0.0
) -> SparsePauliOp:
    """Creates a SparsePauliOp representing a diagonal Hamiltonian in the computational basis.
       The energy of each basis state is derived from its count in the histogram,
       favoring states that meet `min_ones` and are observed, and punishing others.

    Args:
        joint_counts: A Counter object of observed bit string counts.
        num_features: The total number of qubits.
        min_ones: The minimum number of '1's required in a bit string for it to be
                  considered 'favorable' if observed. States not meeting this or unobserved
                  will be assigned `unobserved_punishment` energy.
        unobserved_punishment: The positive energy value assigned to unobserved bit strings,
                               or observed bit strings that don't meet `min_ones` criteria.
                               These states will be 'punished' (avoided by optimizer).
        normalization_offset: An optional offset to subtract from counts before calculating energy.
                              Useful for centering counts, e.g., for 50/50 cases.

    Returns:
        A SparsePauliOp observable.
    """
    
    # 1. Determine the desired energy (E_b) for *each* computational basis state |b>
    # This replaces the old `strength` determination logic.
    state_energies = {} # Maps bit_string -> desired_energy (e.g., "011" -> -500)
    
    # Iterate through all possible bit strings (2^num_features)
    for i in range(2**num_features):
        bit_string = format(i, '0' + str(num_features) + 'b') # Generates MSB...LSB (q_N-1 ... q_0)
        num_ones = bit_string.count('1') # Count '1's for `min_ones` criteria

        if num_ones < min_ones:
            # If it doesn't meet min_ones, treat as undesirable, assign punishment
            energy_b = unobserved_punishment
        elif bit_string in joint_counts:
            # If observed AND meets min_ones: energy proportional to -count
            energy_b = -float(joint_counts[bit_string] - normalization_offset)
        else:
            # If unobserved (but meets min_ones, or min_ones=0): assign punishment
            energy_b = unobserved_punishment
        
        state_energies[bit_string] = energy_b

    # 2. Generate all possible Z-only Pauli strings (e.g., I, Z, II, IZ, ZI, ZZ for 2 qubits)
    # in MSB to LSB order (Q_{N-1}...Q_0) for consistency with Qiskit Pauli string order.
    all_pauli_z_strings = []
    for pauli_tuple in itertools.product('IZ', repeat=num_features):
        pauli_string = "".join(pauli_tuple)
        all_pauli_z_strings.append(pauli_string)
    
    # 3. Convert these state_energies (E_b) into coefficients (c_P) for Pauli strings
    # This loop replaces your original `for i in range(2**num_features)` and `pauli_string` construction.
    pauli_term_coefficients = {}

    for pauli_str in all_pauli_z_strings: # Iterate through all canonical Pauli-Z strings
        coeff_p = 0.0
        # Sum over all basis states 'b' (all 2^N bit strings)
        for bit_string, energy_b in state_energies.items():
            # Get eigenvalue of Pauli string P for basis state |b>
            # <--- CHANGE THIS LINE HERE --- >
            eigenvalue_p_b = _calculate_pauli_z_eigenvalue_for_basis_state(pauli_str, bit_string) 
            coeff_p += energy_b * eigenvalue_p_b
        
        # Normalize by 2^N
        coeff_p /= (2**num_features)
        
        if abs(coeff_p) > 1e-9: # Add term only if coefficient is significant
            pauli_term_coefficients[pauli_str] = coeff_p

    # Convert to SparsePauliOp format
    interaction_strength_list = [(pauli_str, coeff) for pauli_str, coeff in pauli_term_coefficients.items()]
    interaction_observable = SparsePauliOp.from_list(interaction_strength_list)
    return interaction_observable

# You also provided the create_interaction_observable_general function, which is separate
# and was not part of the problem. Keeping it as is.
def create_interaction_observable_general(interactions, num_features):
    """Creates a SparsePauliOp observable for generalized interactions.
    Args:
        interactions: A dictionary where keys are tuples of node indices 
                      (e.g., (0, 1), (0, 0, 2), (0, 1, 2, 3)) and 
                      values are the corresponding interaction strengths.
        num_features: The total number of qubits.

    Returns:
        A SparsePauliOp observable.
    """
    interaction_strength_list = []
    for nodes, strength in interactions.items():
        strength = -strength # Assuming you want to minimize this energy
        pauli_string = ""
        for i in range(num_features):
            if i in nodes:  # Check if the current qubit is in the interaction
                pauli_string += "Z"
            else:
                pauli_string += "I"
        interaction_strength_list.append((pauli_string, strength))

    interaction_observable = SparsePauliOp.from_list(interaction_strength_list)
    return interaction_observable




def create_interaction_observable_general(interactions, num_features):
    """Creates a SparsePauliOp observable for generalized interactions.
    Args:
        interactions: A dictionary where keys are tuples of node indices 
                     (e.g., (0, 1), (0, 0, 2), (0, 1, 2, 3)) and 
                     values are the corresponding interaction strengths.
        num_features: The total number of qubits.

    Returns:
        A SparsePauliOp observable.
    """
    interaction_strength_list = []
    for nodes, strength in interactions.items():
        strength = -strength
        pauli_string = ""
        for i in range(num_features):
            if i in nodes:  # Check if the current qubit is in the interaction
                pauli_string += "Z"
            else:
                pauli_string += "I"
        interaction_strength_list.append((pauli_string, strength))

    interaction_observable = SparsePauliOp.from_list(interaction_strength_list)
    return interaction_observable




def evaluate_and_plot_ansatz(ansatz, params, shots=1024, title="Quantum Sampler Results"):
    """Evaluates a quantum ansatz, plots results and circuit, and prints counts."""
    try:
        sampler = StatevectorSampler()
        bound_circuit = ansatz.copy()
        bound_circuit.assign_parameters(params, inplace=True)
        bound_circuit.measure_all()

        job = sampler.run([bound_circuit], shots=shots)
        pub_result = job.result()[0]
        data_pub = pub_result.data
        counts = data_pub.meas.get_counts()

        print(f"The counts are: {counts}")

        # Plot histogram:
        plot_histogram(counts, bar_labels=True, title=title).show()

        # Matplotlib customization:
        sorted_counts = dict(sorted(counts.items()))
        x_labels = list(sorted_counts.keys())
        y_values = list(sorted_counts.values())

        plt.figure(figsize=(12, 8))
        plt.bar(x_labels, y_values)
        plt.xlabel("Measurement Outcomes", fontsize=16)
        plt.ylabel("Counts", fontsize=16)
        plt.title(title, fontsize=18)
        plt.xticks(rotation=45, ha='right', fontsize=14)
        plt.tight_layout()
        plt.show()

        return counts, bound_circuit

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def create_parameter_dictionaries(combined_qc, ct1_percentages):
    """Creates static and variable parameter dictionaries."""
    # Get Hadamard parameters
    # params_ct = [param for param in combined_qc.parameters if 'ct1_' in param.name or 'ct2_' in param.name
    #                                                         and 'grn' not in param.name 
    #                                                         and 'lr' not in param.name 
    #                                                         and 'act2' not in param.name]
    params_ct = [param for param in combined_qc.parameters if '_act_' in param.name]

    static_params = {}
    for i, val in enumerate(ct1_percentages):
        static_params[params_ct[i]] = val

    variable_params = [param for param in combined_qc.parameters if param not in static_params]
    x0_interaction = np.zeros(len(variable_params))  # All zeros
    variable_params = dict(zip(variable_params, x0_interaction))

    return static_params, variable_params


def create_parameter_dictionaries_cust(combined_qc, ct1_percentages):
    """Creates static and variable parameter dictionaries."""
    params_ct = [param for param in combined_qc.parameters if '_act_' in param.name]

    static_params = {}
    variable_params = [param for param in combined_qc.parameters if param not in static_params]

    # Initialize variable parameters
    #x0_interaction = np.zeros(len(variable_params))  # All zeros
    x0_interaction = np.ones(len(variable_params))*np.pi*0  # All zeros
    variable_params = dict(zip(variable_params, x0_interaction))

    # Now, iterate through the identified 'act' parameters and assign their
    for i, param in enumerate(params_ct):
        variable_params[param] = ct1_percentages[i]

    return static_params, variable_params

# Create the static and variable parameter dictionaries directly from the circuit.
def create_parameter_dictionaries_from_circuit(circuit):
    """Creates static and variable parameter dictionaries directly from the circuit."""
    static_params = {param: None for param in circuit.parameters if 'lr_' not in param.name}
    variable_params = {param: 0.0 for param in circuit.parameters if 'lr_' in param.name}
    return static_params, variable_params

def cost_func_vqe(params, combined_qc, hamiltonian, estimator):  # combined_qc here
    """Cost function for VQE"""
    bound_qc = combined_qc.assign_parameters(params)  # Assign parameters INSIDE cost_func_vqe
    statevector = Statevector(bound_qc)  # Use bound_qc
    statevector_array = statevector.data
    hamiltonian_matrix = hamiltonian.to_matrix()
    energy = np.real(statevector_array.conjugate() @ hamiltonian_matrix @ statevector_array)
    return energy

def cost_func_wrapper(variable_values, all_params, combined_qc, interaction_observable, estimator, variable_params):
    for i, param in enumerate(variable_params):
        all_params[param] = variable_values[i]
    return cost_func_vqe(all_params, combined_qc, interaction_observable, estimator) # Pass combined_qc

import numpy as np
import matplotlib.pyplot as plt
from qiskit.primitives import StatevectorEstimator
from scipy.optimize import minimize
def vqe_solver(
    histogram_data,
    circuit, # Renamed from 'cirquit' for common convention
    ct1_percentages, # Renamed from 'act_percentages' for consistency with create_parameter_dictionaries_cust
    cost_func_wrapper, # This function needs to be defined to accept the correct arguments (see comments below)
    min_ones_obs=1, # Added as an explicit argument for flexibility
    optimizer_method="L-BFGS-B"
):
    """
    Performs a Variational Quantum Eigensolver (VQE) optimization.

    This function encapsulates the VQE workflow, including:
    1. Creating an interaction observable from histogram data.
    2. Setting up static and variable parameters for the quantum circuit.
    3. Initializing the Qiskit StatevectorEstimator.
    4. Running the optimization using scipy.optimize.minimize.
    5. Collecting cost function values during optimization.
    6. Updating parameters with optimized values.
    7. (Optional) Plotting the energy minimization curve.

    Args:
        histogram_data (dict): Data used to create the interaction observable.
        circuit (QuantumCircuit): The parameterized quantum circuit (ansatz).
        ct1_percentages (list): Initial percentage values for parameters
                                containing '_act_' in the circuit.
        cost_func_wrapper (callable): The cost function to be minimized.
                                      It MUST accept arguments in the following order:
                                      cost_func_wrapper(current_variable_params_array,
                                                        static_params_dict,
                                                        circuit,
                                                        observable,
                                                        estimator,
                                                        variable_param_objects_list)
                                      Inside this function, you should combine
                                      static_params_dict and map current_variable_params_array
                                      to variable_param_objects_list to form the
                                      full parameter dictionary for circuit binding.
        min_ones_for_observable (int): Minimum number of '1's for filtering
                                       histogram data when creating the observable.
        optimizer_method (str): The optimization method to use for scipy.minimize
                                (e.g., "L-BFGS-B", "COBYLA", "SLSQP").

    Returns:
        tuple: A tuple containing:
            - result_object (OptimizeResult): The result object from scipy.optimize.minimize.
            - optimized_full_params (dict): The dictionary of all parameters
                                            with their final optimized values.
            - cost_history (list): A list of cost function values recorded
                                   during each optimization iteration.
    """

    num_qubits = circuit.num_qubits # Get number of qubits from the circuit

    # 1. Create interaction observable
    interaction_observable = create_interaction_observable_from_histogram(
        histogram_data, num_qubits, min_ones_obs
    )
    print("Interaction observable CT1 from histogram:", interaction_observable)

    # 2. Create static and variable parameter dictionaries
    # This uses the create_parameter_dictionaries_cust function from the immersive
    # which sets static_params to empty and variable_params to all circuit parameters
    # initialized to 0.0.
    # Ensure you are using create_parameter_dictionaries_cust here, not create_parameter_dictionaries
    # static_params, variable_params_dict = create_parameter_dictionaries_cust(
    #     circuit, ct1_percentages
    # )
    static_params, variable_params_dict = create_parameter_dictionaries(circuit, ct1_percentages) 



    print("Static Parameters:", static_params)
    print("Variable Parameters (initialized to 0.0):", variable_params_dict)

    # 3. Initialize Qiskit Estimator
    estimator = StatevectorEstimator()

    # 4. Prepare initial guess for optimization
    # The 'variable_params_dict' contains parameter objects as keys and 0.0 as values.
    # For scipy.minimize, we need an array of initial values (x0).
    # We extract the initial values from variable_params_dict to form x0.
    x0_interaction = np.array(list(variable_params_dict.values()))

    # To correctly map back optimized values, we also need a list of the parameter objects
    # that correspond to the order of values in x0_interaction.
    variable_param_objects = list(variable_params_dict.keys())


    # 5. Create initial full parameter dictionary (this is mainly for initial print and structure)
    # The actual 'all_params' for binding will be constructed inside cost_func_wrapper.
    initial_all_params_for_display = static_params.copy()
    initial_all_params_for_display.update(dict(zip(variable_param_objects, x0_interaction)))


    # in the context of the vqe_solver function scope.
    iteration_data = {'counter': 0} 
    cost_values = [] # List to store cost values at each iteration
    # Define the callback function for minimize
    def callback_func(xk):
        # Access the counter from the enclosing scope's dictionary
        current_counter = iteration_data['counter']
        if current_counter > 100:
            print_criteria = 100
        else:
            print_criteria = 20

        current_cost = cost_func_wrapper(xk, static_params, circuit, interaction_observable, estimator, variable_param_objects)
        cost_values.append(current_cost)

        # Print the current cost only every 20 iterations
        if current_counter % print_criteria == 0:
            print(f"Iteration {current_counter}: Current cost: {current_cost}")
            
        iteration_data['counter'] += 1 # Increment the counter

    # 6. Call minimize with args
    print(f"Starting optimization with method: {optimizer_method}")
    result_interaction = minimize(
        cost_func_wrapper,
        x0_interaction,
        # IMPORTANT: Pass static_params and variable_param_objects as fixed arguments
        # The cost_func_wrapper will use xk (the first argument) with these to bind parameters.
        args=(static_params, circuit, interaction_observable, estimator, variable_param_objects),
        method=optimizer_method, # Use the passed optimizer_method
        callback=callback_func # Use the defined callback function
        #tol= 1e-5,
    )

    print("\nOptimization Result:")
    print(result_interaction)
    print(f"\nFinal Energy: {result_interaction.fun}")

    # 7. Update the full parameter dictionary with optimized variable parameters
    optimized_variable_parameters = result_interaction.x

    # Construct the final optimized full parameter dictionary
    optimized_full_params = static_params.copy()
    for param_obj, value in zip(variable_param_objects, optimized_variable_parameters):
        optimized_full_params[param_obj] = value

    print("\nOptimized Full Parameters:")
    # Print optimized parameters by name for readability
    for param, value in optimized_full_params.items():
        print(f"  {param.name}: {value}")


    # Return the results
    return result_interaction, optimized_full_params, cost_values