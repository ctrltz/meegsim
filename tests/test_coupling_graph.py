import numpy as np
import networkx as nx
import pytest

from meegsim.coupling_graph import connecting_paths, generate_walkaround_paths


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

    coupling_setup = {
        edge: {
            'method': 'ppc_von_mises', 
            'kappa': kappa_list[i], 
            'phase_lag': phase_lag_list[i]
        } 
        for i, edge in enumerate(edgelist)
    }

    G, walkaround = connecting_paths(coupling_setup, random_state=42)

    assert list(G.edges) == edgelist, "Graph edges do not match the edge list"
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

    coupling_setup = {
        edge: {
            'method': 'ppc_von_mises', 
            'kappa': kappa_list[i], 
            'phase_lag': phase_lag_list[i]
        } 
        for i, edge in enumerate(edgelist)
    }

    with pytest.raises(ValueError, match="The graph contains cycles. Cycles are not supported."):
        connecting_paths(coupling_setup)


def test_connecting_paths_random_state():
    # Test with random_state for reproducibility
    edgelist = [(0, 1), (1, 2), (1, 3)]
    kappa_list = [0.1, 0.2, 0.3]
    phase_lag_list = [0.5, 0.6, 0.7]

    coupling_setup = {
        edge: {
            'method': 'ppc_von_mises', 
            'kappa': kappa_list[i], 
            'phase_lag': phase_lag_list[i]
        } 
        for i, edge in enumerate(edgelist)
    }

    _, walkaround1 = connecting_paths(coupling_setup, random_state=42)
    _, walkaround2 = connecting_paths(coupling_setup, random_state=42)

    assert walkaround1 == walkaround2, "Walkaround paths should be identical with the same random_state"
