import numpy as np
import networkx as nx
from meegsim.coupling_graph import connecting_paths, is_cycle_topology, generate_walkaround_paths


def test_is_cycle_topology_cycle_graph():
    # Create a cycle graph with 5 nodes
    G = nx.cycle_graph(5)
    result = is_cycle_topology(G)
    assert result == True, "Failed on cycle graph: Expected True"


def test_is_cycle_topology_path_graph():
    # Create a path graph with 5 nodes (which is not a cycle)
    G = nx.path_graph(5)
    result = is_cycle_topology(G)
    assert result == False, "Failed on path graph: Expected False"


def test_is_cycle_topology_disconnected_graph():
    # Create a disconnected graph with two components, one of which is a cycle
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0), (3, 4)])
    cycle_component = G.subgraph([0, 1, 2])
    result = is_cycle_topology(cycle_component)
    assert result == True, "Failed on disconnected graph's cycle component: Expected True"

    non_cycle_component = G.subgraph([3, 4])
    result = is_cycle_topology(non_cycle_component)
    assert result == False, "Failed on disconnected graph's non-cycle component: Expected False"


def test_is_cycle_topology_single_node():
    # Test with a single-node graph
    G = nx.Graph()
    G.add_node(0)
    result = is_cycle_topology(G)
    assert result == False, "Failed on single-node graph: Expected False"


def test_is_cycle_topology_empty_graph():
    # Test with an empty graph
    G = nx.Graph()
    result = is_cycle_topology(G)
    assert result == False, "Failed on empty graph: Expected False"


def test_is_cycle_topology_two_node_cycle():
    # Test with a two-node cycle (which technically isn't a cycle in the expected sense)
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 0)])
    result = is_cycle_topology(G)
    assert result == False, "Failed on two-node cycle graph: Expected False"


def test_generate_walkaround_paths_with_start_node():
    # Create a simple tree graph
    tree = nx.Graph()
    tree.add_edges_from([(0, 1), (0, 2), (1, 3), (1, 4)])

    # Generate paths starting from node 0
    result = generate_walkaround_paths(tree, start_node=0)
    expected = [(0, 1), (1, 3), (1, 4), (0, 2)]
    assert result == expected, f"Failed with start_node: Expected {expected}, got {result}"


def test_generate_walkaround_paths_random_start_node():
    # Create a simple tree graph
    tree = nx.Graph()
    tree.add_edges_from([(0, 1), (0, 2), (1, 3), (1, 4)])

    # Generate paths with a random start node using a fixed seed
    result1 = generate_walkaround_paths(tree, random_state=42)
    result2 = generate_walkaround_paths(tree, random_state=42)
    assert result1 == result2, "Failed with random_state: Results should be identical"


def test_generate_walkaround_paths_single_node():
    # Test with a single-node tree
    tree = nx.Graph()
    tree.add_node(0)

    # Generate paths starting from node 0
    result = generate_walkaround_paths(tree, start_node=0)
    expected = []
    assert result == expected, f"Failed on single-node tree: Expected {expected}, got {result}"


def test_connecting_paths_tree_topology():
    # Test with a simple tree topology
    edgelist = [(0, 1), (1, 2), (1, 3)]
    kappa_list = [0.1, 0.2, 0.3]
    phase_lag_list = [0.5, 0.6, 0.7]

    G, walkaround = connecting_paths(edgelist, kappa_list, phase_lag_list, random_state=42)

    assert len(G.edges) == len(edgelist), "Graph edges do not match the edge list"
    assert len(walkaround) == 1, "There should be one walkaround path for one tree topology"

    # Convert lists to sets of frozensets to account for order invariance
    expected_walkaround = {(0, 1), (1, 2), (1, 3)}
    result_walkaround = set(walkaround[0])

    assert result_walkaround == expected_walkaround, (
        f"Expected walkaround paths {expected_walkaround}, got {result_walkaround}"
    )


def test_connecting_paths_with_cycle_topology():
    # Test with a graph containing a cycle
    edgelist = [(0, 1), (1, 2), (2, 0)]
    kappa_list = [0.1, 0.2, 0.3]
    phase_lag_list = [0.5, 0.6, 0.7]

    try:
        connecting_paths(edgelist, kappa_list, phase_lag_list)
    except ValueError as e:
        assert str(e) == "The graph contains cycles.", f"Unexpected error message: {e}"


def test_connecting_paths_random_state():
    # Test with random_state for reproducibility
    edgelist = [(0, 1), (1, 2), (1, 3)]
    kappa_list = [0.1, 0.2, 0.3]
    phase_lag_list = [0.5, 0.6, 0.7]

    G1, walkaround1 = connecting_paths(edgelist, kappa_list, phase_lag_list, random_state=42)
    G2, walkaround2 = connecting_paths(edgelist, kappa_list, phase_lag_list, random_state=42)

    assert walkaround1 == walkaround2, "Walkaround paths should be identical with the same random_state"

