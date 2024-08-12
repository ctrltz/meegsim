import re
import numpy as np
import mne
import pytest

from mock import patch

from meegsim.location import select_random
from meegsim.utils import unpack_vertices


def create_dummy_sourcespace(vertices):
    # Fill in dummy data as a constant time series equal to the vertex number
    n_src_spaces = len(vertices)
    type_src = 'surf' if n_src_spaces == 2 else 'vol'
    src = []
    for i in range(n_src_spaces):
        # Create a simple dummy data structure
        n_verts = len(vertices[i])
        vertno = vertices[i]  # Vertices for this hemisphere
        xyz = np.random.rand(n_verts, 3) * 100  # Random positions

        # Explicitly set types to match src objects that are created by MNE
        src_dict = dict(
            vertno=np.array(vertno),
            rr=np.array(xyz),
            nn=np.random.rand(n_verts, 3),  # Random normals
            inuse=np.ones(n_verts, dtype=int),  # All vertices in use
            nuse=int(n_verts),
            type=str(type_src),
            id=int(i),
            np=int(n_verts)
        )
        src.append(src_dict)

    return mne.SourceSpaces(src)


def test_single_space_basic_functionality():
    # Test basic functionality with a single source space
    vertices = [[0, 1, 2, 3, 4]]
    single_src = create_dummy_sourcespace(vertices)
    result = select_random(single_src, n=2, random_state=42)
    assert len(result) == 2, f"Expected 2 vertices, got {len(result)}"
    assert all(vert[1] in single_src[0]['vertno'] for vert in result), "Selected vertices are not in the source space"


def test_dual_space_basic_functionality():
    # Test basic functionality with two source spaces
    vertices = [[0, 1, 2], [3, 4, 5]]
    dual_src = create_dummy_sourcespace(vertices)
    result = select_random(dual_src, n=2, random_state=42)
    assert len(result) == 2, f"Expected 2 vertices, got {len(result)}"
    assert all(vert in unpack_vertices([list(s['vertno']) for s in dual_src]) for vert in result), "Selected vertices are not in the source spaces"


def test_specific_vertices():
    # Test selection from a specific set of vertices
    vertices = [[0, 1, 2, 3, 4, 5, 6]]
    single_src = create_dummy_sourcespace(vertices)
    specific_vertices = [[1, 2, 3]]
    result = select_random(single_src, vertices=specific_vertices, n=1, random_state=42)
    assert len(result) == 1, f"Expected 1 vertex, got {len(result)}"
    assert all(vert in unpack_vertices(specific_vertices) for vert in result), "Selected vertex is not in the specific set"


def test_random_state_effect():
    # Test that the same random_state produces the same result
    vertices = [[0, 1, 2, 3, 4, 5, 6]]
    single_src = create_dummy_sourcespace(vertices)
    result1 = select_random(single_src, n=3, random_state=42)
    result2 = select_random(single_src, n=3, random_state=42)
    assert result1 == result2, "Results with the same random state should be identical"


def test_invalid_vertices_error():
    # Test error when invalid vertices are provided
    vertices = [[0, 1, 2, 3, 4]]
    single_src = create_dummy_sourcespace(vertices)
    with pytest.raises(ValueError, match="Some vertices are not contained in the src."):
        select_random(single_src, vertices=[[5, 6]], n=1)


def test_invalid_source_space_length():
    # Test error for incorrect source space length
    with pytest.raises(ValueError, match=re.escape("Src must contain either one (volume) or two (surface) source spaces.")):
        select_random([], n=1)


def test_more_than_available_vertices():
    # Test selecting more vertices than available
    vertices = [[0, 1, 2, 3, 4]]
    single_src = create_dummy_sourcespace(vertices)
    with pytest.raises(ValueError, match="Number of vertices to select exceeds available vertices."):
        select_random(single_src, n=6)


def test_select_random_sort_output():
    # Create a mock that returns some unsorted vertices
    # Sorting should reverse the list
    initial = [(1, 2), (1, 0), (0, 1)]
    expected = initial[::-1]

    class MockGenerator:
        def choice(self, a, size=None, replace=False):
            # NB: rng.choice always returns an array
            # rows corresponds to different elements
            return np.array(initial)

    vertices = [[0, 1], [0, 1, 2]]
    src = create_dummy_sourcespace(vertices)

    # Replace the numpy generator with our mock and check the sorting
    with patch('numpy.random.default_rng') as mock_rng:
        mock_rng.return_value = MockGenerator()
        assert select_random(src, n=3, sort_output=True) == expected
