import numpy as np
import mne

from meegsim.utils import combine_stcs, normalize_power


def prepare_stc(vertices, num_samples=5):
    # Fill in dummy data as a constant time series equal to the vertex number
    data = np.tile(vertices[0] + vertices[1], reps=(num_samples, 1)).T
    return mne.SourceEstimate(data, vertices, tmin=0, tstep=0.01)


def test_combine_stcs_no_overlap():
    vertices1 = [[1, 3, 5], [4, 5]]
    vertices2 = [[2, 4], [1, 2, 3]]

    stc1 = prepare_stc(vertices1)
    stc2 = prepare_stc(vertices2)

    # Vertices should be sorted in the combined stc, and the
    # corresponding time series should be in the correct order
    expected_vertices = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
    expected_data = np.tile([1, 2, 3, 4, 5, 1, 2, 3, 4, 5], reps=(5, 1)).T

    stc = combine_stcs(stc1, stc2)
    assert np.array_equal(stc.vertices, expected_vertices)
    assert np.array_equal(stc.data, expected_data)


def test_combine_stcs_overlap():
    vertices1 = [[1, 2], [3]]
    vertices2 = [[2, 3], [1, 2, 3]]

    stc1 = prepare_stc(vertices1)
    stc2 = prepare_stc(vertices2)

    # Vertices should be sorted in the combined stc
    expected_vertices = [[1, 2, 3], [1, 2, 3]]

    # The signals corresponding to overlapping vertices should be
    # summed (2 + 2 = 4 in the left and 3 + 3 = 6 in the right hemisphere)
    expected_data = np.tile([1, 4, 3, 1, 2, 6], reps=(5, 1)).T

    stc = combine_stcs(stc1, stc2)
    assert np.array_equal(stc.vertices, expected_vertices)
    assert np.array_equal(stc.data, expected_data)


def test_normalize_power():
    data = np.random.randn(10, 1000)
    normalized = normalize_power(data)

    # Should not change the shape but should change the norm
    assert data.shape == normalized.shape
    assert np.allclose(np.linalg.norm(normalized, axis=1), 1)