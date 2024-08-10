import numpy as np
import mne
import pytest

from meegsim.utils import unpack_vertices


def test_unpack_single_list():
    vertices_lists = [[1, 2, 3]]
    expected_output = [(0, 1), (0, 2), (0, 3)]
    assert unpack_vertices(vertices_lists) == expected_output


def test_unpack_multiple_lists():
    vertices_lists = [[1, 2], [3, 4]]
    expected_output = [(0, 1), (0, 2), (1, 3), (1, 4)]
    assert unpack_vertices(vertices_lists) == expected_output


def test_unpack_mixed_empty_non_empty_lists():
    vertices_lists = [[1, 2], []]
    expected_output = [(0, 1), (0, 2)]
    assert unpack_vertices(vertices_lists) == expected_output


def test_unpack_warning_for_single_list():
    with pytest.warns(UserWarning, match="Input is not a list of lists. Will be assumed that there is one source space."):
        result = unpack_vertices([1, 2, 3])
    expected_output = [(0, 1), (0, 2), (0, 3)]
    assert result == expected_output


def test_unpack_repeated_vertices():
    vertices_lists = [[1, 1, 2], [3, 3, 4]]
    expected_output = [(0, 1), (0, 1), (0, 2), (1, 3), (1, 3), (1, 4)]
    assert unpack_vertices(vertices_lists) == expected_output

