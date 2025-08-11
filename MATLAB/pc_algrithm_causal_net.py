import numpy as np
import pandas as pd
from itertools import combinations
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple, Set, Dict, Optional

class PCAlgorithm:
    """
    Implementation of the PC (Peter-Clark) algorithm for causal discovery.
    
    The PC algorithm learns causal structure through:
    1. Skeleton discovery using conditional independence tests
    2. Edge orientation using v-structures and propagation rules
    """
    
    def __init__(self, alpha: float = 0.05, max_conditioning_set_size: int = 3):
        """
        Initialize PC algorithm.
        
        Args:
            alpha: Significance level for independence tests
            max_conditioning_set_size: Maximum size of conditioning sets to consider
        """
        self.alpha = alpha
        self.max_k = max_conditioning_set_size
        self.skeleton = None
        self.separating_sets = {}
        self.directed_graph = None
        
    def fit(self, data: pd.DataFrame) -> nx.DiGraph:
        """
        Learn causal structure from data.
        
        Args:
            data: DataFrame with variables as columns
            
        Returns:
            Directed graph representing learned causal structure
        """
        self.variables = list(data.columns)
        self.n_vars = len(self.variables)
        self.data = data
        
        # Phase 1: Learn skeleton
        print("Phase 1: Learning skeleton...")
        self.skeleton = self._learn_skeleton()
        
        # Phase 2: Orient edges
        print("Phase 2: Orienting edges...")
        self.directed_graph = self._orient_edges()
        
        return self.directed_graph
    
    def _learn_skeleton(self) -> nx.Graph:
        """Learn the skeleton (undirected graph) using conditional independence tests."""
        # Start with complete graph
        skeleton = nx.complete_graph(self.variables)
        self.separating_sets = {}
        
        # Test conditional independence for increasing conditioning set sizes
        for k in range(self.max_k + 1):
            print(f"  Testing with conditioning sets of size {k}")
            edges_to_remove = []
            
            for edge in list(skeleton.edges()):
                x, y = edge
                
                # Get potential conditioning sets (neighbors of x and y, excluding x and y)
                neighbors_x = set(skeleton.neighbors(x)) - {y}
                neighbors_y = set(skeleton.neighbors(y)) - {x}
                potential_z = neighbors_x.union(neighbors_y)
                
                # Test all conditioning sets of size k
                if len(potential_z) >= k:
                    for z_set in combinations(potential_z, k):
                        if self._test_conditional_independence(x, y, list(z_set)):
                            edges_to_remove.append((x, y))
                            self.separating_sets[(x, y)] = list(z_set)
                            self.separating_sets[(y, x)] = list(z_set)
                            break
            
            # Remove edges found to be conditionally independent
            skeleton.remove_edges_from(edges_to_remove)
            
        return skeleton
    
    def _test_conditional_independence(self, x: str, y: str, z: List[str]) -> bool:
        """
        Test conditional independence between x and y given z.
        
        Uses partial correlation for continuous variables.
        """
        if len(z) == 0:
            # Test marginal independence
            return self._test_marginal_independence(x, y)
        
        # Calculate partial correlation
        partial_corr = self._partial_correlation(x, y, z)
        
        # Fisher's z-transformation for significance test
        n = len(self.data)
        z_score = 0.5 * np.log((1 + partial_corr) / (1 - partial_corr)) * np.sqrt(n - len(z) - 3)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        return p_value > self.alpha
    
    def _test_marginal_independence(self, x: str, y: str) -> bool:
        """Test marginal independence between x and y."""
        corr, p_value = pearsonr(self.data[x], self.data[y])
        return p_value > self.alpha
    
    def _partial_correlation(self, x: str, y: str, z: List[str]) -> float:
        """Calculate partial correlation between x and y given z."""
        if len(z) == 0:
            return pearsonr(self.data[x], self.data[y])[0]
        
        # Create design matrix
        vars_subset = [x, y] + z
        data_subset = self.data[vars_subset].values
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(data_subset.T)
        
        # Calculate partial correlation using matrix inversion
        try:
            inv_corr = np.linalg.inv(corr_matrix)
            partial_corr = -inv_corr[0, 1] / np.sqrt(inv_corr[0, 0] * inv_corr[1, 1])
            return partial_corr
        except np.linalg.LinAlgError:
            # Fallback to simple correlation if matrix is singular
            return pearsonr(self.data[x], self.data[y])[0]
    
    def _orient_edges(self) -> nx.DiGraph:
        """Orient edges in the skeleton to form a directed acyclic graph."""
        # Convert skeleton to directed graph (with both directions for each edge)
        directed = nx.DiGraph()
        for edge in self.skeleton.edges():
            directed.add_edge(edge[0], edge[1])
            directed.add_edge(edge[1], edge[0])
        
        # Rule 1: Orient v-structures (colliders)
        self._orient_v_structures(directed)
        
        # Rules 2-4: Apply Meek's orientation rules
        changed = True
        while changed:
            changed = False
            changed |= self._meek_rule_1(directed)
            changed |= self._meek_rule_2(directed)
            changed |= self._meek_rule_3(directed)
        
        return directed
    
    def _orient_v_structures(self, graph: nx.DiGraph):
        """Identify and orient v-structures (X -> Z <- Y where X and Y are not adjacent)."""
        for z in self.variables:
            # Find all pairs of non-adjacent neighbors of z
            neighbors = list(self.skeleton.neighbors(z))
            
            for x, y in combinations(neighbors, 2):
                # Check if x and y are not adjacent in skeleton
                if not self.skeleton.has_edge(x, y):
                    # Check if z is not in the separating set of x and y
                    sep_set = self.separating_sets.get((x, y), [])
                    if z not in sep_set:
                        # Orient as v-structure: x -> z <- y
                        if graph.has_edge(z, x):
                            graph.remove_edge(z, x)
                        if graph.has_edge(z, y):
                            graph.remove_edge(z, y)
    
    def _meek_rule_1(self, graph: nx.DiGraph) -> bool:
        """
        Rule 1: If X -> Y - Z and X and Z not adjacent, then orient Y -> Z.
        """
        changed = False
        for y in self.variables:
            # Find directed edges into y
            parents_y = [x for x in graph.predecessors(y) if not graph.has_edge(y, x)]
            # Find undirected edges from y
            undirected_from_y = [z for z in graph.successors(y) if graph.has_edge(z, y)]
            
            for x in parents_y:
                for z in undirected_from_y:
                    # Check if x and z are not adjacent
                    if not self.skeleton.has_edge(x, z):
                        # Orient y -> z
                        if graph.has_edge(z, y):
                            graph.remove_edge(z, y)
                            changed = True
        
        return changed
    
    def _meek_rule_2(self, graph: nx.DiGraph) -> bool:
        """
        Rule 2: If X -> Y -> Z and X - Z, then orient X -> Z.
        """
        changed = False
        for y in self.variables:
            parents_y = [x for x in graph.predecessors(y) if not graph.has_edge(y, x)]
            children_y = [z for z in graph.successors(y) if not graph.has_edge(z, y)]
            
            for x in parents_y:
                for z in children_y:
                    # Check if x - z (undirected)
                    if graph.has_edge(x, z) and graph.has_edge(z, x):
                        # Orient x -> z
                        graph.remove_edge(z, x)
                        changed = True
        
        return changed
    
    def _meek_rule_3(self, graph: nx.DiGraph) -> bool:
        """
        Rule 3: If X - Y, X - Z, Y -> W, Z -> W, and Y and Z not adjacent, then X -> Y.
        """
        changed = False
        for x in self.variables:
            # Find undirected neighbors of x
            undirected_neighbors = [n for n in graph.successors(x) if graph.has_edge(n, x)]
            
            for y, z in combinations(undirected_neighbors, 2):
                # Check if y and z are not adjacent
                if not self.skeleton.has_edge(y, z):
                    # Find common children of y and z
                    children_y = set(n for n in graph.successors(y) if not graph.has_edge(n, y))
                    children_z = set(n for n in graph.successors(z) if not graph.has_edge(n, z))
                    common_children = children_y.intersection(children_z)
                    
                    if common_children:
                        # Orient x -> y and x -> z
                        if graph.has_edge(y, x):
                            graph.remove_edge(y, x)
                            changed = True
                        if graph.has_edge(z, x):
                            graph.remove_edge(z, x)
                            changed = True
        
        return changed
    
    def plot_results(self, figsize: Tuple[int, int] = (12, 5)):
        """Plot both skeleton and final directed graph."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot skeleton
        pos = nx.spring_layout(self.skeleton, seed=42)
        nx.draw(self.skeleton, pos, ax=ax1, with_labels=True, node_color='lightblue',
                node_size=1000, font_size=10, font_weight='bold')
        ax1.set_title("Learned Skeleton")
        
        # Plot directed graph
        nx.draw(self.directed_graph, pos, ax=ax2, with_labels=True, node_color='lightcoral',
                node_size=1000, font_size=10, font_weight='bold', arrows=True,
                arrowsize=20, arrowstyle='->')
        ax2.set_title("Final Causal Graph")
        
        plt.tight_layout()
        plt.show()

# Example usage and testing
def generate_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate sample data with known causal structure: A -> B -> C, A -> D -> C"""
    np.random.seed(42)
    
    # Generate data according to: A -> B -> C, A -> D -> C
    A = np.random.normal(0, 1, n_samples)
    B = 0.8 * A + np.random.normal(0, 0.5, n_samples)
    D = 0.6 * A + np.random.normal(0, 0.5, n_samples)
    C = 0.7 * B + 0.5 * D + np.random.normal(0, 0.3, n_samples)
    
    return pd.DataFrame({
        'A': A,
        'B': B, 
        'C': C,
        'D': D
    })

if __name__ == "__main__":
    # Generate sample data
    print("Generating sample data with structure: A -> B -> C, A -> D -> C")
    data = generate_sample_data()
    
    # Run PC algorithm
    pc = PCAlgorithm(alpha=0.05, max_conditioning_set_size=2)
    learned_graph = pc.fit(data)
    
    # Print results
    print(f"\nLearned {learned_graph.number_of_edges()} directed edges:")
    for edge in learned_graph.edges():
        print(f"  {edge[0]} -> {edge[1]}")
    
    # Plot results
    pc.plot_results()
    
    # Print adjacency matrix
    print(f"\nAdjacency matrix of learned graph:")
    adj_matrix = nx.adjacency_matrix(learned_graph, nodelist=sorted(pc.variables))
    print(pd.DataFrame(adj_matrix.toarray(), 
                      columns=sorted(pc.variables), 
                      index=sorted(pc.variables)))