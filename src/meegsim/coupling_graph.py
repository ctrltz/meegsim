import networkx as nx
import numpy as np

from .utils import get_sfreq


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


def generate_walkaround(coupling_graph, random_state=None):
    """
    Constructs a graph from the provided edge list and attributes, and identifies walkaround paths in tree topologies.

    Parameters
    ----------
    coupling_graph : nx.Graph
        The coupling graph that describes the desired connectivity patterns.
        All edges should have the coupling parameters as attributes.
    random_state : int or None, optional
        Seed for the random number generator. If start_node is None, the start node will be drawn
        randomly, and results will vary between function calls. default = None.

    Returns
    -------
    walkaround : list of tuples
        A list of coupling edges (source, target) ordered in a way that guarantees the
        desired coupling for all the edges.
    """

    if not nx.is_forest(coupling_graph):
        raise ValueError("The graph contains cycles. Cycles are not supported.")

    # iterate over connected components
    walkaround = []
    for component in nx.connected_components(coupling_graph):
        subgraph = coupling_graph.subgraph(component)

        # build the path starting from random node
        walkaround_paths = traverse_tree(
            subgraph, start_node=None, random_state=random_state
        )
        walkaround.extend(walkaround_paths)

    return walkaround


def _set_coupling(sources, coupling_graph, times, random_state):
    """
    This function traverses the coupling graph and executes the simulation
    of coupling for each edge in the graph.

    Parameters
    ----------
    sources : dict
        Simulated sources.
    coupling_graph : nx.Graph
        The coupling graph that describes the desired connectivity pattern.
    times : array-like
        The time points for all samples in the waveform.
    random_state : int or None
        The random state that could be fixed to ensure reproducibility.
    """
    walkaround = generate_walkaround(coupling_graph, random_state=random_state)

    for name1, name2 in walkaround:
        # Get the sources by their names
        s1, s2 = sources[name1], sources[name2]

        # Get the corresponding coupling parameters
        coupling_params = coupling_graph.get_edge_data(name1, name2)

        # Extract the coupling method temporarily
        tmp_coupling_params = coupling_params.copy()
        coupling_fn = tmp_coupling_params.pop("method")

        # Adjust the waveform of s2 to be coupled with s1
        s2.waveform = coupling_fn(
            s1.waveform,
            get_sfreq(times),
            **tmp_coupling_params,
            random_state=random_state,
        )
