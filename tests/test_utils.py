import numpy as np
import mne
import pytest

from mne.io.constants import FIFF
from meegsim.utils import (
    _extract_hemi,
    unpack_vertices,
    combine_stcs,
    normalize_variance,
    get_sfreq,
    vertices_to_mne,
    _hemi_to_index,
    _get_param_from_stc,
    _get_center_of_mass,
)

from utils.prepare import prepare_source_space, prepare_source_estimate


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
    with pytest.warns(
        UserWarning,
        match="Input is not a list of lists. Will be assumed that there is one source space.",
    ):
        result = unpack_vertices([1, 2, 3])
    expected_output = [(0, 1), (0, 2), (0, 3)]
    assert result == expected_output


def test_unpack_repeated_vertices():
    vertices_lists = [[1, 1, 2], [3, 3, 4]]
    expected_output = [(0, 1), (0, 1), (0, 2), (1, 3), (1, 3), (1, 4)]
    assert unpack_vertices(vertices_lists) == expected_output


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


def test_normalize_variance():
    data = np.random.randn(10, 1000)
    normalized = normalize_variance(data)

    # Should not change the shape but should change the norm
    assert data.shape == normalized.shape
    assert np.allclose(np.var(normalized, axis=1), 1)


def test_get_sfreq():
    sfreq = 250
    times = np.arange(0, sfreq) / sfreq
    assert get_sfreq(times) == sfreq


def test_get_sfreq_too_few_timepoints_raises():
    with pytest.raises(ValueError, match="must contain at least two points"):
        get_sfreq(np.array([0]))


def test_get_sfreq_unequal_spacing_raises():
    with pytest.raises(ValueError, match="not uniformly spaced"):
        get_sfreq(np.array([0, 0.01, 0.1]))


def test_extract_hemi():
    src = mne.SourceSpaces(
        [
            {"type": "surf", "id": FIFF.FIFFV_MNE_SURF_LEFT_HEMI},
            {"type": "surf", "id": FIFF.FIFFV_MNE_SURF_RIGHT_HEMI},
            {"type": "vol", "id": FIFF.FIFFV_MNE_SURF_UNKNOWN},
            {"type": "discrete", "id": FIFF.FIFFV_MNE_SURF_UNKNOWN},
        ]
    )
    expected_hemis = ["lh", "rh", None, None]

    for s, hemi in zip(src, expected_hemis):
        assert _extract_hemi(s) == hemi, f"Failed for {s['type']}"


def test_extract_hemi_raises():
    src = [
        {"id": 0},  # no 'type'
        {"type": "vol"},  # no 'id'
    ]

    for s in src:
        with pytest.raises(ValueError, match="mandatory internal fields"):
            _extract_hemi(s)

    src = [{"type": "surf", "id": FIFF.FIFFV_MNE_SURF_UNKNOWN}]

    with pytest.raises(ValueError, match="Unexpected ID"):
        _extract_hemi(src[0])


def test_vertices_to_mne():
    src = prepare_source_space(["surf", "surf"], [[0, 1, 2], [0, 1, 2]])

    packed = vertices_to_mne([(0, 0)], src)
    assert packed == [[0], []]

    # vertices should be sorted
    packed = vertices_to_mne([(0, 2), (0, 0)], src)
    assert packed == [[0, 2], []]

    packed = vertices_to_mne([(0, 0), (1, 2)], src)
    assert packed == [[0], [2]]

    packed = vertices_to_mne([(1, 0), (1, 2)], src)
    assert packed == [[], [0, 2]]


def test_hemi_to_index():
    assert _hemi_to_index("lh") == 0
    assert _hemi_to_index("rh") == 1


def test_get_param_from_stc_lh():
    stc = prepare_source_estimate(data=[0, 1, 0, 0], vertices=[[0, 1], [0, 1]])
    assert _get_param_from_stc(stc, [(0, 1)]) == 1


def test_get_param_from_stc_rh():
    stc = prepare_source_estimate(data=[0, 0, 0, 1], vertices=[[0, 1], [0, 1]])
    assert _get_param_from_stc(stc, [(1, 1)]) == 1


def test_get_param_from_stc_multiple():
    stc = prepare_source_estimate(data=[0, 1, 2, 3], vertices=[[0, 1], [0, 1]])
    all_vertices = [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert np.allclose(_get_param_from_stc(stc, all_vertices), np.arange(4))


def test_get_param_from_stc_duplicate_vertex():
    stc = prepare_source_estimate(data=[0, 0, 0, 1], vertices=[[0, 1], [0, 1]])

    # Request the same vertex multiple times
    all_vertices = [(1, 1), (1, 1), (1, 1)]
    assert np.allclose(_get_param_from_stc(stc, all_vertices), np.ones((3,)))


def test_get_center_of_mass_one_vertex():
    src = prepare_source_space(["surf", "surf"], [[0, 1, 2], [0, 1, 2]])
    for v in [0, 1, 2]:
        assert _get_center_of_mass(src, 0, [v]) == v
        assert _get_center_of_mass(src, 1, [v]) == v


def test_get_center_of_mass_multiple_vertices():
    src = prepare_source_space(["surf", "surf"], [[0, 1, 2, 3, 4], [0, 1, 2]])
    src[0]["rr"] = np.tile(np.array([[-2], [-1], [0], [1], [2]]), (1, 3))
    print(src[0]["rr"])
    assert _get_center_of_mass(src, 0, [0, 1, 2, 3, 4]) == 2
