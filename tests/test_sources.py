from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from meegsim.sources import (
    _BaseSource,
    PointSource,
    PatchSource,
    _combine_sources_into_stc,
    _get_point_sources_in_hemi,
    _get_patch_sources_in_hemis,
)

from utils.prepare import prepare_source_space, prepare_source_estimate


def test_basesource_is_abstract():
    waveform = np.ones((100,))
    s = _BaseSource(0, waveform)
    with pytest.raises(NotImplementedError, match="in a subclass"):
        s.data


# =================================
# Point source
# =================================


@pytest.mark.parametrize(
    "src_idx,vertno,hemi",
    [
        (0, 123, None),
        (0, 234, "lh"),
        (1, 345, None),
        (1, 456, "rh"),
    ],
)
def test_pointsource_repr(src_idx, vertno, hemi):
    # Waveform is not required for repr, leaving it empty
    s = PointSource("mysource", src_idx, vertno, np.array([]), hemi=hemi)

    if hemi is None:
        assert f"src[{src_idx}]" in repr(s)
    else:
        assert hemi in repr(s)

    assert str(vertno) in repr(s)
    assert "mysource" in repr(s)


@pytest.mark.parametrize("src_idx,hemi", [(0, "lh"), (1, "rh")])
def test_pointsource_to_label_should_pass(src_idx, hemi):
    src = prepare_source_space(types=["surf", "surf"], vertices=[[0, 1], [0, 1]])
    s = PointSource("mysource", src_idx, 1, np.array([]), hemi=hemi)
    label = s.to_label(src)

    assert np.isin(1, label.vertices)
    assert label.hemi == hemi


def test_pointsource_to_label_bad_source_space_type():
    src = prepare_source_space(
        types=["surf", "surf", "vol"], vertices=[[0, 1], [0, 1], [0, 1, 2]]
    )
    s = PointSource("mysource", 2, 1, np.array([]))
    with pytest.raises(ValueError, match="Only sources in surface"):
        s.to_label(src)


@pytest.mark.parametrize("src_idx,vertno", [(0, 0), (1, 1)])
def test_pointsource_to_stc(src_idx, vertno):
    waveform = np.ones((100,))
    src = prepare_source_space(types=["surf", "surf"], vertices=[[0, 1], [0, 1]])
    s = PointSource("mysource", src_idx, vertno, waveform)
    stc = s.to_stc(src, tstep=0.01)

    assert (
        stc.data.shape[0] == 1
    ), f"Expected one active vertex in stc, got {stc.data.shape[0]}"
    assert (
        vertno in stc.vertices[src_idx]
    ), f"Expected the vertex to be put in src {src_idx}, but it is not there"
    assert np.allclose(
        stc.data, waveform
    ), "The source waveform should not change during conversion to stc"


@pytest.mark.parametrize("tstep", [0.01, 0.025, 0.05])
def test_pointsource_to_stc_tstep(tstep):
    waveform = np.ones((100,))
    src = prepare_source_space(types=["surf", "surf"], vertices=[[0, 1], [0, 1]])
    s = PointSource("mysource", 0, 0, waveform)
    stc = s.to_stc(src, tstep=tstep)

    expected_sfreq = 1.0 / tstep
    assert (
        stc.sfreq == expected_sfreq
    ), f"Expected stc.sfreq to be {expected_sfreq}, got {stc.sfreq}"


def test_pointsource_to_stc_subject():
    waveform = np.ones((100,))
    src = prepare_source_space(types=["surf", "surf"], vertices=[[0, 1], [0, 1]])
    s = PointSource("mysource", 0, 0, waveform)
    stc = s.to_stc(src, tstep=0.01)

    assert (
        stc.subject == "meegsim"
    ), f"Expected stc.subject to be derived from src, got {stc.subject}"

    stc = s.to_stc(src, tstep=0.01, subject="mysubject")

    assert (
        stc.subject == "mysubject"
    ), f"Expected stc.subject to be mysubject, got {stc.subject}"


def test_pointsource_to_stc_bad_src_raises():
    waveform = np.ones((100,))
    src = prepare_source_space(types=["surf", "surf"], vertices=[[0, 1], [0, 1]])

    # src[2] is out of range
    s = PointSource("mysource", 2, 0, waveform)
    with pytest.raises(ValueError, match="point source was assigned to source space 2"):
        s.to_stc(src, tstep=0.01, subject="mysubject")


def test_pointsource_to_stc_bad_vertno_raises():
    waveform = np.ones((100,))
    src = prepare_source_space(types=["surf", "surf"], vertices=[[0, 1], [0, 1]])

    # vertex 2 is not in src[0]
    s = PointSource("mysource", 0, 2, waveform)
    with pytest.raises(ValueError, match="contain the following vertices: 2"):
        s.to_stc(src, tstep=0.01, subject="mysubject")


def test_pointsource_create_from_arrays():
    n_sources = 2
    n_samples = 1000
    sfreq = 250

    src = prepare_source_space(types=["surf", "surf"], vertices=[[0, 1], [0, 1]])
    times = np.arange(n_samples) / sfreq
    location = [(0, 0), (1, 1)]
    waveform = np.tile(np.arange(n_sources), (n_samples, 1)).T
    stds = [1, 2]
    names = ["s1", "s2"]

    sources = PointSource._create(
        src, times, n_sources, location, waveform, stds, names
    )

    # Check that the inputs were distributed correctly between sources
    assert [s.src_idx for s in sources] == [0, 1]
    assert [s.vertno for s in sources] == [0, 1]
    assert [s.waveform[0] for s in sources] == [0, 1]
    assert [s.std for s in sources] == stds
    assert [s.name for s in sources] == names


def test_pointsource_create_from_callables():
    n_sources = 2
    n_samples = 1000
    sfreq = 250

    def location_pick_first(src, random_state=None):
        return [(idx, s["vertno"][0]) for idx, s in enumerate(src)]

    def waveform_constant(n_sources, times, random_state=None):
        return np.tile(np.arange(n_sources), (len(times), 1)).T

    src = prepare_source_space(types=["surf", "surf"], vertices=[[0, 1], [0, 1]])
    times = np.arange(n_samples) / sfreq
    location = location_pick_first
    waveform = waveform_constant
    stds = [1, 2]
    names = ["s1", "s2"]

    sources = PointSource._create(
        src, times, n_sources, location, waveform, stds, names
    )

    # Check that the inputs were distributed correctly between sources
    assert [s.src_idx for s in sources] == [0, 1]
    assert [s.vertno for s in sources] == [0, 0]
    assert [s.waveform[0] for s in sources] == [0, 1]
    assert [s.std for s in sources] == stds
    assert [s.name for s in sources] == names


@patch("meegsim.sources._get_param_from_stc", return_value=np.array([1, 2]))
def test_pointsource_create_std_sourceestimate(get_param_mock):
    n_sources = 2
    n_samples = 1000
    sfreq = 250

    vertices = [[0, 1], [0, 1]]
    src = prepare_source_space(types=["surf", "surf"], vertices=vertices)
    times = np.arange(n_samples) / sfreq
    location = [(0, 0), (1, 1)]
    waveform = np.tile(np.arange(n_sources), (n_samples, 1)).T
    std_stc = prepare_source_estimate(data=[0, 1, 2, 3], vertices=vertices)
    stds = [1, 2]
    names = ["s1", "s2"]

    # Values are passed directly - the mock should not be used
    sources = PointSource._create(
        src, times, n_sources, location, waveform, stds, names
    )
    get_param_mock.assert_not_called()

    # Values are passed in stc - the mock should be called once
    sources = PointSource._create(
        src, times, n_sources, location, waveform, std_stc, names
    )
    get_param_mock.assert_called_once()
    assert [s.std for s in sources] == [1, 2]


# =================================
# Patch source
# =================================


@pytest.mark.parametrize(
    "src_idx,vertno,hemi",
    [
        (0, [123], None),
        (0, [123, 234], "lh"),
        (1, [345], None),
        (1, [123, 234, 345, 456], "rh"),
    ],
)
def test_patchsource_repr(src_idx, vertno, hemi):
    # Waveform is not required for repr, leaving it empty
    s = PatchSource("mysource", src_idx, vertno, np.array([]), hemi=hemi)

    if hemi is None:
        assert f"src[{src_idx}]" in repr(s)
    else:
        assert hemi in repr(s)

    assert str(len(vertno)) in repr(s)
    assert "mysource" in repr(s)


@pytest.mark.parametrize("src_idx,hemi", [(0, "lh"), (1, "rh")])
def test_patchsource_to_label_should_pass(src_idx, hemi):
    src = prepare_source_space(types=["surf", "surf"], vertices=[[0, 1], [0, 1]])
    s = PatchSource("mysource", src_idx, [0, 1], np.array([]), hemi=hemi)
    label = s.to_label(src)

    assert np.isin(0, label.vertices)
    assert np.isin(1, label.vertices)
    assert label.hemi == hemi


@pytest.mark.parametrize("src_idx,vertno", [(0, [0, 1]), (1, [1, 2])])
def test_patchsource_to_stc(src_idx, vertno):
    # adjust the waveform to account for scaling in .data
    waveform = np.ones((100,)) * np.sqrt(len(vertno))
    src = prepare_source_space(types=["surf", "surf"], vertices=[[0, 1], [0, 1, 2]])
    s = PatchSource("mysource", src_idx, vertno, waveform)
    stc = s.to_stc(src, tstep=0.01)
    n_vertno = len(vertno)

    assert (
        stc.data.shape[0] == n_vertno
    ), f"Expected {n_vertno} active vertices in stc, got {stc.data.shape[0]}"
    assert np.all(
        vertno in stc.vertices[src_idx]
    ), f"Expected all vertno to be put in src {src_idx}"
    assert np.allclose(
        stc.data, 1.0
    ), "The source waveform should be scaled by the square root of the number of vertices"


def test_patchsource_to_stc_bad_src_raises():
    waveform = np.ones((100,))
    src = prepare_source_space(types=["surf", "surf"], vertices=[[0, 1], [0, 1]])

    # src[2] is out of range
    s = PatchSource("mysource", 2, [0, 1], waveform)
    with pytest.raises(ValueError, match="patch source was assigned to source space 2"):
        s.to_stc(src, tstep=0.01, subject="mysubject")


def test_patchsource_to_stc_bad_vertno_raises():
    waveform = np.ones((100,))
    src = prepare_source_space(types=["surf", "surf"], vertices=[[0, 1], [0, 1]])

    # vertex 2 is not in src[0]
    s = PatchSource("mysource", 0, [0, 2], waveform)
    with pytest.raises(ValueError, match="does not contain the following vertices: 2"):
        s.to_stc(src, tstep=0.01, subject="mysubject")


def test_patchsource_create_with_extent():
    """Test that PatchSource properly handles 'extent' parameter."""

    src = prepare_source_space(
        types=["surf", "surf"], vertices=[[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
    )

    # Mock the times array
    n_samples = 100
    sfreq = 250
    times = np.arange(n_samples) / sfreq

    # Number of sources
    n_sources = 2

    # Mock location and waveform
    location = [(0, 2), (1, 8)]  # (src_idx, vertno)
    waveform = np.array(
        [
            np.ones(
                n_samples,
            ),
            np.ones(
                n_samples,
            ),
        ]
    )  # Two waveforms for two sources
    stds = [1, 2]
    names = ["source_1", "source_2"]

    # Mock extents for each source
    extents = [3, None]  # First source has extent, second has no extent

    # Mock grow_labels return values (for the first source with extent)
    vertno = [2, 3, 4]  # Vertices grown for the first source

    with patch("mne.grow_labels") as mock_grow_labels:
        # Mock grow_labels to return the desired vertices
        mock_grow_labels.return_value = [MagicMock(vertices=vertno)]

        # Call the create method
        sources = PatchSource._create(
            src=src,
            times=times,
            n_sources=n_sources,
            location=location,
            waveform=waveform,
            stds=stds,
            names=names,
            extents=extents,
            subject=None,
            subjects_dir=None,
            random_state=None,
        )

        # Check that two sources were created
        assert len(sources) == 2, "The number of created sources is incorrect"

        # Check the first source (with extent)
        source_1 = sources[0]
        assert source_1.name == "source_1", "First source name mismatch"
        assert source_1.src_idx == 0, "First source src_idx mismatch"
        assert source_1.vertno == vertno, "First source vertno mismatch"
        assert source_1.std == 1, "First source std mismatch"

        # Check the second source (without extent)
        source_2 = sources[1]
        assert source_2.name == "source_2", "Second source name mismatch"
        assert source_2.src_idx == 1, "Second source src_idx mismatch"
        assert source_2.vertno == [8], "Second source vertno mismatch"
        assert source_2.std == 2, "Second source std mismatch"

        # Verify that grow_labels was called once for the source with extent
        mock_grow_labels.assert_called_once_with(
            subject="meegsim", seeds=[2], extents=3, hemis=0, subjects_dir=None
        )


@patch("meegsim.sources._get_param_from_stc", side_effect=[1, 4])
def test_patchsource_create_std_sourceestimate(get_param_mock):
    n_sources = 2
    n_samples = 1000
    sfreq = 250

    vertices = [[0, 1], [0, 1]]
    src = prepare_source_space(types=["surf", "surf"], vertices=vertices)
    times = np.arange(n_samples) / sfreq
    location = [(0, [0, 1]), (1, [0, 1])]
    waveform = np.tile(np.arange(n_sources), (n_samples, 1)).T
    std_stc = prepare_source_estimate(data=[0, 1, 2, 3], vertices=vertices)
    stds = [1, 2]
    names = ["s1", "s2"]
    extents = [None] * n_sources

    # Values are passed directly - the mock should not be used
    sources = PatchSource._create(
        src, times, n_sources, location, waveform, stds, names, extents, None, None
    )
    get_param_mock.assert_not_called()

    # Values are passed in stc - the mock should be called once per patch
    sources = PatchSource._create(
        src, times, n_sources, location, waveform, std_stc, names, extents, None, None
    )
    assert get_param_mock.call_count == n_sources

    # Mock std values should be saved
    assert np.allclose([s.std for s in sources], [1, 4])


# =================================
# Helper functions
# =================================


def test_combine_sources_into_stc_point():
    src = prepare_source_space(types=["surf", "surf"], vertices=[[0, 1], [0, 1]])

    s1 = PointSource("s1", 0, 0, np.ones((100,)))
    s2 = PointSource("s2", 0, 0, np.ones((100,)))
    s3 = PointSource("s3", 0, 1, np.ones((100,)))

    # s1 and s2 are the same vertex, should be summed
    stc1 = _combine_sources_into_stc([s1, s2], src, tstep=0.01)
    assert stc1.data.shape[0] == 1, "Expected 1 active vertices in stc"
    assert np.all(stc1.data == 2), "Expected source activity to be summed"

    # s1 and s3 are different vertices, should be concatenated
    stc2 = _combine_sources_into_stc([s1, s3], src, tstep=0.01)
    assert stc2.data.shape[0] == 2, "Expected 2 active vertices in stc"
    assert np.all(stc2.data == 1), "Expected source activity not to be summed"


def test_combine_sources_into_stc_patch():
    src = prepare_source_space(types=["surf", "surf"], vertices=[[0, 1], [0, 1]])

    # adjust the waveform to account for scaling in .data
    s1 = PatchSource("s1", 0, [0, 1], np.ones((100,)) * np.sqrt(2))
    s2 = PatchSource("s2", 1, [0, 1], np.ones((100,)) * np.sqrt(2))
    s3 = PatchSource("s3", 0, [0, 1], np.ones((100,)) * np.sqrt(2))

    # s1 and s2 are in different hemispheres, activity should not be summed
    stc1 = _combine_sources_into_stc([s1, s2], src, tstep=0.01)
    assert stc1.data.shape[0] == 4, "Expected 4 active vertices in stc"
    assert np.allclose(stc1.data, 1.0), "Expected source activity not to be summed"

    # s1 and s3 are in the same hemisphere, activity should be summed
    stc2 = _combine_sources_into_stc([s1, s3], src, tstep=0.01)
    assert stc2.data.shape[0] == 2, "Expected 2 active vertices in stc"
    assert np.allclose(stc2.data, 2.0), "Expected source activity to be summed"


def test_get_point_sources_in_hemi():
    sources = [
        # left hemisphere
        PointSource("s1", 0, 0, []),
        PointSource("s2", 0, 1, []),
        # right hemisphere
        PointSource("s3", 1, 2, []),
        PointSource("s4", 1, 3, []),
    ]

    assert _get_point_sources_in_hemi(sources, "lh") == [0, 1]
    assert _get_point_sources_in_hemi(sources, "rh") == [2, 3]


def test_get_patch_sources_in_hemis_no_patches():
    src = prepare_source_space(["surf", "surf"], [[0, 1], [0, 1]])
    sources = [
        PointSource("s1", 0, 0, []),
        PointSource("s2", 0, 1, []),
    ]

    stc = _get_patch_sources_in_hemis(sources, src, ["lh", "rh"])
    assert np.allclose(stc.data, 0, atol=0.1)  # ignore 0.01s


def test_get_patch_sources_in_hemis():
    src = prepare_source_space(["surf", "surf"], [[0, 1, 2, 3], [0, 1, 2, 3]])
    sources = [
        PatchSource("s1", 0, [0], []),
        PatchSource("s2", 1, [1, 2, 3], []),
    ]

    stc_both = _get_patch_sources_in_hemis(sources, src, ["lh", "rh"])
    assert stc_both.data.size == 8
    assert np.allclose(stc_both.data.sum(), 4, atol=0.1)  # ignore 0.01s

    stc_lh = _get_patch_sources_in_hemis(sources, src, ["lh"])
    assert stc_lh.data.size == 8
    assert np.allclose(stc_lh.data.sum(), 1, atol=0.1)  # ignore 0.01s

    stc_rh = _get_patch_sources_in_hemis(sources, src, ["rh"])
    assert stc_rh.data.size == 8
    assert np.allclose(stc_rh.data.sum(), 3, atol=0.1)  # ignore 0.01s
