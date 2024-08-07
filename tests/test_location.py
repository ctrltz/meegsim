import numpy as np
from meegsim.location import select_random


def test_single_space_basic_functionality():
    # Test basic functionality with a single source space
    single_src = [{'vertno': np.array([0, 1, 2, 3, 4])}]
    result = select_random(single_src, n=2, random_state=42)
    assert len(result) == 2, f"Expected 2 vertices, got {len(result)}"
    assert all(vert in single_src[0]['vertno'] for vert in result), "Selected vertices are not in the source space"


def test_dual_space_basic_functionality():
    # Test basic functionality with two source spaces
    dual_src = [
        {'vertno': np.array([0, 1, 2]), 'np': 2},
        {'vertno': np.array([3, 4, 5]), 'np': 2}
    ]
    lh_result, rh_result = select_random(dual_src, n=2, random_state=42)
    combined_result = lh_result + rh_result
    assert len(combined_result) == 2, f"Expected 2 vertices, got {len(combined_result)}"
    expected_vertices = np.concatenate([dual_src[0]['vertno'], dual_src[1]['vertno'] + dual_src[1]['np']])
    assert all(vert in expected_vertices for vert in combined_result), "Selected vertices are not in the source spaces"


def test_specific_vertices():
    # Test selection from a specific set of vertices
    single_src = [{'vertno': np.array([0, 1, 2, 3, 4, 5, 6])}]
    specific_vertices = np.array([1, 2, 3])
    result = select_random(single_src, vertices=specific_vertices, n=1, random_state=42)
    assert len(result) == 1, f"Expected 1 vertex, got {len(result)}"
    assert result[0] in specific_vertices, "Selected vertex is not in the specific set"


def test_random_state_effect():
    # Test that the same random_state produces the same result
    single_src = [{'vertno': np.array([0, 1, 2, 3, 4, 5, 6])}]
    result1 = select_random(single_src, n=3, random_state=42)
    result2 = select_random(single_src, n=3, random_state=42)
    assert result1.tolist() == result2.tolist(), "Results with the same random state should be identical"


def test_invalid_vertices_error():
    # Test error when invalid vertices are provided
    single_src = [{'vertno': np.array([0, 1, 2, 3, 4])}]
    try:
        select_random(single_src, vertices=np.array([5, 6]), n=1)
        print("test_invalid_vertices_error failed: Expected ValueError")
    except ValueError as e:
        print("test_invalid_vertices_error passed")

def test_invalid_source_space_length():
    # Test error for incorrect source space length
    try:
        select_random([], n=1)
        print("test_invalid_source_space_length failed: Expected ValueError")
    except ValueError as e:
        print("test_invalid_source_space_length passed")

def test_more_than_available_vertices():
    # Test selecting more vertices than available
    single_src = [{'vertno': np.array([0, 1, 2, 3, 4])}]
    try:
        select_random(single_src, n=6)
        print("test_more_than_available_vertices failed: Expected ValueError")
    except ValueError as e:
        assert str(e) == "Number of vertices to select exceeds available vertices."

