import networkx as nx
import numpy as np


def generate_walkaround_paths(tree, start_node=None, random_state=None):
    """
    Generate a list of walkaround paths in a tree starting from start_node.

    Walkaround paths are pairs of nodes where each pair represents an edge
    in the tree, starting from the specified start_node.

    Parameters:
    ----------
    tree : networkx.Graph
        The tree in which to generate walkaround paths.
    start_node : int
        The node from which to start generating paths. 
        If start_node is None (default), the start node will be drawn randomly.
    random_state : int or None, optional
        Seed for the random number generator. If start_node is None (default), the 
        start node will be drawn randomly, and results will vary between function calls.

    Returns:
    -------
    out : list of lists of int
        A list of pairs of nodes representing walkaround paths.
    """

    if start_node is None:
        # take random
        rng = np.random.default_rng(random_state)
        start_node = rng.choice(list(tree.nodes))

    return list(nx.dfs_edges(tree, source=start_node))


def connecting_paths(coupling_setup, random_state=None):
    """
    Constructs a graph from the provided edge list and attributes, and identifies walkaround paths in tree topologies.

    Parameters
    ----------
    coupling_setup : dict
        with keys being edges (source, target)
        with values being coupling parameters dict(method='ppc_von_mises', kappa=0.5, phase_lag=1)
    random_state : int or None, optional
        Seed for the random number generator. If start_node is None, the start node will be drawn
        randomly, and results will vary between function calls. default = None.    
        
    Returns
    -------
    out : tuple (G, walkaround)
        - G : networkx.Graph
            The constructed graph with edges, weights, and capacities.
        - walkaround : list of lists
            A list of walkaround paths for each tree topology in the graph. Each walkaround path is a list of node pairs.
    """

    # Build graph
    G = nx.Graph()
    G.add_edges_from(coupling_setup)

    if not nx.is_forest(G):
        raise ValueError("The graph contains cycles. Cycles are not supported.")

    # iterate over connected components
    walkaround = []
    for component in nx.connected_components(G):
        subgraph = G.subgraph(component)

        # build the path starting from random node
        walkaround_paths = generate_walkaround_paths(subgraph, start_node=None, 
                                                     random_state=random_state)
        walkaround.append(walkaround_paths)

    return G, walkaround
