import networkx as nx
import numpy as np

from .coupling import _coupling_dispatcher


def traverse_tree(tree, start_node=None, random_state=None):
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
    out : list of tuples
        A list of pairs of nodes representing walkaround paths.
    """

    if start_node is None:
        # take random
        rng = np.random.default_rng(random_state)
        start_node = rng.choice(list(tree.nodes))

    return list(nx.dfs_edges(tree, source=start_node))


def generate_walkaround(coupling_setup, random_state=None):
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
    walkaround : list of tuples
        A list of coupling edges (source, target) ordered in a way that guarantees the 
        desired coupling for all the edges.
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
        walkaround_paths = traverse_tree(subgraph, start_node=None, 
                                         random_state=random_state)
        walkaround.extend(walkaround_paths)

    return walkaround


def _set_coupling(sources, coupling, times, random_state):
    """
    This function traverses the coupling graph and executes the simulation
    of coupling for each edge in the graph.

    Parameters
    ----------
    sources: dict
        Simulated sources.
    coupling: dict.
        The coupling to be added.
    times: array-like
        The time points for all samples in the waveform.
    random_state: int or None
        The random state that could be fixed to ensure reproducibility.

    Returns
    -------
    sources: dict
        Simulated sources with waveforms adjusted according to the desired coupling.
    """
    walkaround = generate_walkaround(coupling, random_state=random_state)

    for name1, name2 in walkaround:
        # Get the sources by their names
        s1, s2 = sources[name1], sources[name2]
        
        # Get the corresponding coupling parameters
        # NOTE: for now, we assume undirected connectivity, so the edges might
        # get reversed during walkaround (i.e., (0, 1) was defined but (1, 0)
        # was required for the correct traversal of the coupling graph).
        # 
        # As a temporary fix, we restore the original order here. 
        # A long-term solution should address the directed vs. undirected type 
        # of connectivity more specifically for built-in functions as well.
        edge = (name1, name2)
        if edge not in coupling:
            edge = (name2, name1)
        coupling_params = coupling[edge]

        # Adjust the waveform of s2 to be coupled with s1
        s2.waveform = _coupling_dispatcher(s1.waveform, coupling_params,
                                           times, random_state=random_state)

    return sources