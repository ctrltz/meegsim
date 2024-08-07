import random
import networkx as nx


def is_cycle_topology(component):
    """
    Check if a given connected component of a graph is a cycle topology.

    A cycle topology is defined as a connected component where:
    - Every node in the component has exactly two neighbors (degree of 2).
    - The component contains exactly one cycle.

    Parameters:
    ----------
    component : networkx.Graph
        A connected component of the graph to check.

    Returns:
    -------
    out : bool
        True if the component is a cycle topology, False otherwise.
    """
    # Check if every node in the component has a degree of 2
    degrees = dict(component.degree())
    if all(degree == 2 for degree in degrees.values()):
        # Verify if the component is indeed a cycle
        # A cycle will have exactly one cycle in its cycle_basis
        return len(nx.cycle_basis(component)) == 1
    return False

def find_cycle_topologies(graph):
    """
    Identify and return all cycle topologies within a graph.

    Parameters:
    ----------
    graph : networkx.Graph
        The graph to analyze.

    Returns:
    -------
    out : list of networkx.Graph
        A list of subgraphs, each representing a cycle topology.
    """
    cycle_topologies = []
    for component in nx.connected_components(graph):
        subgraph = graph.subgraph(component)
        if is_cycle_topology(subgraph):
            cycle_topologies.append(subgraph)
    return cycle_topologies

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


def connecting_paths(edgelist, kappa_list, phase_lag_list, random_state=None):
    """
    Constructs a graph from the provided edge list and attributes, and identifies walkaround paths in tree topologies.

    Parameters:
    -------
    edgelist : list of tuples
        A list of edges where each edge is represented as a tuple (node1, node2).
    kappa_list : list of float
        A list of weights corresponding to each edge in `edgelist`.
    phase_lag_list : list of float
        A list of capacities corresponding to each edge in `edgelist`.
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
    for i_edge in range(len(edgelist)):
        G.add_edge(edgelist[i_edge][0], edgelist[i_edge][1], weight=kappa_list[i_edge], capacity=phase_lag_list[i_edge])

    # find tree topologies
    tree_topologies = find_tree_topologies(G)
    if len(tree_topologies) < len(list(nx.connected_components(G))):
        cycle_topologies = find_cycle_topologies(G)
        if len(cycle_topologies) > 0:
            raise ValueError("The graph contains cycles.")
        else:
            raise ValueError("There is some unknown topology in the graph.")

    # iterate over tree_topologies
    walkaround = []
    for tree_topology in tree_topologies:
        # build the path starting from random node
        walkaround.append(generate_walkaround_paths(tree_topology, start_node=None, random_state=random_state))

    return G, walkaround

