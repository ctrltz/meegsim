import random
import networkx as nx


def find_tree_topologies(graph):
    """
    Identify and return all tree topologies within a graph.

    Parameters:
    ----------
    graph : networkx.Graph
        The graph to analyze.

    Returns:
    -------
    out : list of networkx.Graph
        A list of subgraphs, each representing a tree topology.
    """
    tree_topologies = []
    for component in nx.connected_components(graph):
        subgraph = graph.subgraph(component)
        if nx.is_tree(subgraph):
            tree_topologies.append(subgraph)
    return tree_topologies

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
        The node from which to start generating paths. If start_node is None, the start node will be drawn
        randomly. default=None
    random_state : int or None, optional
        Seed for the random number generator. If start_node is None, the start node will be drawn
        randomly, and results will vary between function calls. default = None.

    Returns:
    -------
    out : list of lists of int
        A list of pairs of nodes representing walkaround paths.
    """
    if start_node is None:
        # take random
        random.seed(random_state)
        start_node = random.choice(list(tree.nodes))

    return list(nx.dfs_edges(tree, source=start_node))


def connecting_paths(coupling_setup, random_state=None):
    """
    Constructs a graph from the provided edge list and attributes, and identifies walkaround paths in tree topologies.

    Parameters:
    -------
    coupling_setup : dict
        with keys being edges (source, target)
        with values being coupling parameters dict(method='ppc_von_mises', kappa=0.5, phase_lag=1)
    random_state : int or None, optional
        Seed for the random number generator. If start_node is None, the start node will be drawn
        randomly, and results will vary between function calls. default = None.    Returns:
    -------
    out : tuple (G, walkaround)
        - G : networkx.Graph
            The constructed graph with edges, weights, and capacities.
        - walkaround : list of lists
            A list of walkaround paths for each tree topology in the graph. Each walkaround path is a list of node pairs.
    """
    # Build graph
    G = nx.Graph()
    for node_pair, coupling_param in coupling_setup.items():
        if coupling_param['method'] == 'ppc_von_mises':
            G.add_edge(node_pair[0], node_pair[1], weight=coupling_param['kappa'], capacity=coupling_param['phase_lag'])
        elif coupling_param['method'] == 'constant_phase_shift':
            G.add_edge(node_pair[0], node_pair[1], capacity=coupling_param['phase_lag'])

    if not nx.is_forest(G):
        raise ValueError("The graph contains cycles. Cycles are not supported.")

    # find tree topologies
    tree_topologies = find_tree_topologies(G)

    # iterate over tree_topologies
    walkaround = []
    for tree_topology in tree_topologies:
        # build the path starting from random node
        walkaround.append(generate_walkaround_paths(tree_topology, start_node=None, random_state=random_state))

    return G, walkaround

