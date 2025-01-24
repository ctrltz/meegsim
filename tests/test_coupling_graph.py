import numpy as np
import networkx as nx
import pytest

from mock import patch

from meegsim.coupling_graph import generate_walkaround, traverse_tree, _set_coupling

from utils.prepare import prepare_point_source


def test_traverse_tree_with_start_node():
    # Create a simple tree graph
    tree = nx.Graph()
    tree.add_edges_from([(0, 1), (0, 2), (1, 3), (1, 4)])

    # Generate paths starting from node 0
    result = traverse_tree(tree, start_node=0)
    expected = [(0, 1), (1, 3), (1, 4), (0, 2)]
    assert (
        result == expected
    ), f"Failed with start_node: Expected {expected}, got {result}"


def test_traverse_tree_random_start_node():
    # Create a simple tree graph
    tree = nx.Graph()
    tree.add_edges_from([(0, 1), (0, 2), (1, 3), (1, 4)])

    # Generate paths with a random start node using a fixed seed
    result1 = traverse_tree(tree, random_state=42)
    result2 = traverse_tree(tree, random_state=42)
    assert result1 == result2, "Failed with random_state: Results should be identical"


def test_traverse_tree_single_node():
    # Test with a single-node tree
    tree = nx.Graph()
    tree.add_node(0)

    # Generate paths starting from node 0
    result = traverse_tree(tree, start_node=0)
    expected = []
    assert (
        result == expected
    ), f"Failed on single-node tree: Expected {expected}, got {result}"


def test_generate_walkaround():
    # Test with a simple topology with two trees
    # The result should be a flat list even if several connected components
    # are present in the coupling graph
    edgelist = [(0, 1), (1, 2), (1, 3), (4, 5), (5, 6), (6, 7)]
    kappa_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    phase_lag_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1]

    coupling_setup = {
        edge: {
            "method": "ppc_von_mises",
            "kappa": kappa_list[i],
            "phase_lag": phase_lag_list[i],
        }
        for i, edge in enumerate(edgelist)
    }
    coupling_graph = nx.Graph()
    coupling_graph.add_edges_from(coupling_setup)

    walkaround = generate_walkaround(coupling_graph, random_state=42)
    assert set(walkaround) == set(
        edgelist
    ), "All edges should be included in the walkaround"


def test_generate_walkaround_with_cycle():
    # Test with a graph containing a cycle
    edgelist = [(0, 1), (1, 2), (2, 0)]
    kappa_list = [0.1, 0.2, 0.3]
    phase_lag_list = [0.5, 0.6, 0.7]

    coupling_setup = {
        edge: {
            "method": "ppc_von_mises",
            "kappa": kappa_list[i],
            "phase_lag": phase_lag_list[i],
        }
        for i, edge in enumerate(edgelist)
    }
    coupling_graph = nx.Graph()
    coupling_graph.add_edges_from(coupling_setup)

    with pytest.raises(
        ValueError, match="The graph contains cycles. Cycles are not supported."
    ):
        generate_walkaround(coupling_graph)


def test_generate_walkaround_random_state():
    # Test with random_state for reproducibility
    edgelist = [(0, 1), (1, 2), (1, 3)]
    kappa_list = [0.1, 0.2, 0.3]
    phase_lag_list = [0.5, 0.6, 0.7]

    coupling_setup = {
        edge: {
            "method": "ppc_von_mises",
            "kappa": kappa_list[i],
            "phase_lag": phase_lag_list[i],
        }
        for i, edge in enumerate(edgelist)
    }
    coupling_graph = nx.Graph()
    coupling_graph.add_edges_from(coupling_setup)

    walkaround1 = generate_walkaround(coupling_graph, random_state=42)
    walkaround2 = generate_walkaround(coupling_graph, random_state=42)

    assert (
        walkaround1 == walkaround2
    ), "Walkaround paths should be identical with the same random_state"


@patch(
    "meegsim.coupling_graph.generate_walkaround",
    return_value=[("s1", "s2"), ("s2", "s3")],
)
def test_set_coupling(generate_mock):
    def coupling_fn(waveform, sfreq, kappa, random_state=0):
        side_effect = [1, 2]
        return (kappa, side_effect[kappa])

    sources = {k: prepare_point_source(name=k) for k in ["s1", "s2", "s3"]}
    coupling = [
        ("s1", "s2", dict(method=coupling_fn, kappa=0)),
        # this edge is reversed in the mock walkaround
        ("s3", "s2", dict(method=coupling_fn, kappa=1)),
    ]

    coupling_graph = nx.Graph()
    coupling_graph.add_edges_from(coupling)
    times = np.arange(100) / 100

    _set_coupling(sources, coupling_graph, times, random_state=0)

    # Check that the coupled waveforms were saved correctly
    # First value in the tuple is the provided argument for kappa
    assert sources["s2"].waveform == (0, 1)
    assert sources["s3"].waveform == (1, 2)


@patch("meegsim.coupling_graph.generate_walkaround", return_value=[("s1", "s2")])
def test_set_coupling_random_state(generate_mock):
    def coupling_fn(waveform, sfreq, random_state=0):
        return random_state

    sources = {k: prepare_point_source(name=k) for k in ["s1", "s2", "s3"]}
    coupling = [("s1", "s2", dict(method=coupling_fn))]
    coupling_graph = nx.Graph()
    coupling_graph.add_edges_from(coupling)
    times = np.arange(100) / 100
    random_state = 1234

    _set_coupling(sources, coupling_graph, times, random_state)

    # Check that the random state was forwarded to low-level functions
    assert generate_mock.call_args.kwargs["random_state"] == random_state

    # coupling_fn returns the provided random_state so it should be
    # saved in the source waveform
    assert sources["s2"].waveform == random_state
