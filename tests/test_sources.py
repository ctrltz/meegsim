import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from meegsim.sources import _BaseSource, PointSource, _combine_sources_into_stc, PatchSource

from utils.prepare import prepare_source_space


def test_basesource_is_abstract():
    waveform = np.ones((100,))
    s = _BaseSource(waveform, sfreq=250)
    with pytest.raises(NotImplementedError, match="in the subclass"):
        s.to_stc()



# =================================
# Point source
# =================================
@pytest.mark.parametrize(
    "src_idx,vertno,hemi", [
        (0, 123, None),
        (0, 234, 'lh'),
        (1, 345, None),
        (1, 456, 'rh'),
    ]
)
def test_pointsource_repr(src_idx, vertno, hemi):
    # Waveform is not required for repr, leaving it empty
    s = PointSource('mysource', src_idx, vertno, np.array([]), sfreq=250, hemi=hemi)

    if hemi is None:
        assert f'src[{src_idx}]' in repr(s)
    else:
        assert hemi in repr(s)

    assert str(vertno) in repr(s)
    assert 'mysource' in repr(s)


@pytest.mark.parametrize(
    "src_idx,vertno", [
        (0, 0),
        (1, 1)
    ]
)
def test_pointsource_to_stc(src_idx, vertno):
    waveform = np.ones((100,))
    src = prepare_source_space(
        types=['surf', 'surf'],
        vertices=[[0, 1], [0, 1]]
    )
    s = PointSource('mysource', src_idx, vertno, waveform, sfreq=100)
    stc = s.to_stc(src)

    assert stc.data.shape[0] == 1, \
        f"Expected one active vertex in stc, got {stc.data.shape[0]}"
    assert vertno in stc.vertices[src_idx], \
        f"Expected the vertex to be put in src {src_idx}, but it is not there"
    assert np.allclose(stc.data, waveform), \
        f"The source waveform should not change during conversion to stc"
    

@pytest.mark.parametrize("sfreq", [100, 250, 500])
def test_pointsource_to_stc_sfreq(sfreq):
    waveform = np.ones((100,))
    src = prepare_source_space(
        types=['surf', 'surf'],
        vertices=[[0, 1], [0, 1]]
    )
    s = PointSource('mysource', 0, 0, waveform, sfreq=sfreq)
    stc = s.to_stc(src)

    assert stc.sfreq == sfreq, \
        f"Expected stc.sfreq to be {sfreq}, got {stc.sfreq}"
    

def test_pointsource_to_stc_subject():
    waveform = np.ones((100,))
    src = prepare_source_space(
        types=['surf', 'surf'],
        vertices=[[0, 1], [0, 1]]
    )
    s = PointSource('mysource', 0, 0, waveform, sfreq=250)
    stc = s.to_stc(src)

    assert stc.subject == 'meegsim', \
        f"Expected stc.subject to be derived from src, got {stc.subject}"
    
    stc = s.to_stc(src, subject='mysubject')

    assert stc.subject == 'mysubject', \
        f"Expected stc.subject to be mysubject, got {stc.subject}"
    

def test_pointsource_to_stc_bad_src_raises():
    waveform = np.ones((100,))
    src = prepare_source_space(
        types=['surf', 'surf'],
        vertices=[[0, 1], [0, 1]]
    )

    # src[2] is out of range
    s = PointSource('mysource', 2, 0, waveform, sfreq=250)
    with pytest.raises(ValueError, match="not present in the provided src"):
        s.to_stc(src, subject='mysubject')


def test_pointsource_to_stc_bad_vertno_raises():
    waveform = np.ones((100,))
    src = prepare_source_space(
        types=['surf', 'surf'],
        vertices=[[0, 1], [0, 1]]
    )

    # vertex 2 is not in src[0]
    s = PointSource('mysource', 0, 2, waveform, sfreq=250)
    with pytest.raises(ValueError, match="does not contain the vertex"):
        s.to_stc(src, subject='mysubject')


def test_pointsource_create_from_arrays():
    n_sources = 2
    n_samples = 1000
    sfreq = 250

    src = prepare_source_space(
        types=['surf', 'surf'],
        vertices=[[0, 1], [0, 1]]
    )
    times = np.arange(n_samples) / sfreq
    location = [(0, 0), (1, 1)]
    waveform = np.tile(np.arange(n_sources), (n_samples, 1)).T
    names = ['s1', 's2']

    sources = PointSource.create(src, times, n_sources, location, waveform, names)

    # Check that the inputs were distributed correctly between sources
    assert [s.src_idx for s in sources] == [0, 1]
    assert [s.vertno for s in sources] == [0, 1]
    assert [s.waveform[0] for s in sources] == [0, 1]
    assert [s.name for s in sources] == names


def test_pointsource_create_from_callables():
    n_sources = 2
    n_samples = 1000
    sfreq = 250

    def location_pick_first(src, random_state=None):
        return [(idx, s['vertno'][0]) for idx, s in enumerate(src)]

    def waveform_constant(n_sources, times, random_state=None):
        return np.tile(np.arange(n_sources), (len(times), 1)).T
    
    src = prepare_source_space(
        types=['surf', 'surf'],
        vertices=[[0, 1], [0, 1]]
    )
    times = np.arange(n_samples) / sfreq
    location = location_pick_first
    waveform = waveform_constant
    names = ['s1', 's2']

    sources = PointSource.create(src, times, n_sources, location, waveform, names)

    # Check that the inputs were distributed correctly between sources
    assert [s.src_idx for s in sources] == [0, 1]
    assert [s.vertno for s in sources] == [0, 0]
    assert [s.waveform[0] for s in sources] == [0, 1]
    assert [s.name for s in sources] == names


def test_combine_sources_into_stc():
    src = prepare_source_space(
        types=['surf', 'surf'],
        vertices=[[0, 1], [0, 1]]
    )

    s1 = PointSource('s1', 0, 0, np.ones((100,)), 250)
    s2 = PointSource('s2', 0, 0, np.ones((100,)), 250)
    s3 = PointSource('s3', 0, 1, np.ones((100,)), 250)

    # s1 and s2 are the same vertex, should be summed
    stc1 = _combine_sources_into_stc([s1, s2], src)
    assert stc1.data.shape[0] == 1, 'Expected 1 active vertices in stc'
    assert np.all(stc1.data == 2), 'Expected source activity to be summed'

    # s1 and s3 are different vertices, should be concatenated
    stc2 = _combine_sources_into_stc([s1, s3], src)
    assert stc2.data.shape[0] == 2, 'Expected 2 active vertices in stc'
    assert np.all(stc2.data == 1), 'Expected source activity not to be summed'


# =================================
# Patch source
# =================================
@pytest.mark.parametrize(
    "src_idx,vertno,patch_vertno,hemi", [
        (0, 123, [123], None),
        (0, 234, [234], 'lh'),
        (1, 345, [345], None),
        (1, 456, [456], 'rh'),
    ]
)
def test_patchsource_repr(src_idx, vertno, patch_vertno, hemi):
    # Waveform is not required for repr, leaving it empty
    s = PatchSource('mysource', src_idx, vertno, patch_vertno, np.array([]), sfreq=250, extent=1, hemi=hemi)

    if hemi is None:
        assert f'src[{src_idx}]' in repr(s)
    else:
        assert hemi in repr(s)

    assert str(vertno) in repr(s)
    assert str(patch_vertno) in repr(s)
    assert 'mysource' in repr(s)


@pytest.mark.parametrize(
    "src_idx,vertno,patch_vertno", [
        (0, 0, [0, 1]),
        (1, 1, [1, 2])
    ]
)
def test_patchsource_to_stc(src_idx, vertno, patch_vertno):
    waveform = np.ones((100,))
    src = prepare_source_space(
        types=['surf', 'surf'],
        vertices=[[0, 1], [0, 1]]
    )
    s = PatchSource('mysource', src_idx, vertno, patch_vertno, waveform, sfreq=100, extent=1)
    stc = s.to_stc(src)

    assert stc.data.shape[0] == 2, \
        f"Expected one active vertex in stc, got {stc.data.shape[0]}"
    assert vertno in stc.vertices[src_idx], \
        f"Expected the vertex to be put in src {src_idx}, but it is not there"
    assert np.allclose(stc.data, waveform), \
        f"The source waveform should not change during conversion to stc"


def test_patchsource_to_stc_bad_src_raises():
    waveform = np.ones((100,))
    src = prepare_source_space(
        types=['surf', 'surf'],
        vertices=[[0, 1], [0, 1]]
    )

    # src[2] is out of range
    s = PatchSource('mysource', 2, 0, [0, 1], waveform, sfreq=250, extent=1)
    with pytest.raises(ValueError, match="not present in the provided src"):
        s.to_stc(src, subject='mysubject')


def test_patchsource_to_stc_bad_vertno_raises():
    waveform = np.ones((100,))
    src = prepare_source_space(
        types=['surf', 'surf'],
        vertices=[[0, 1], [0, 1]]
    )

    # vertex 2 is not in src[0]
    s = PatchSource('mysource', 0, 2, [0, 2], waveform, sfreq=250, extent=1)
    with pytest.raises(ValueError, match="does not contain the vertex"):
        s.to_stc(src, subject='mysubject')


def test_combine_patchsources_into_stc():
    src = prepare_source_space(
        types=['surf', 'surf'],
        vertices=[[0, 1], [0, 1]]
    )

    s1 = PatchSource('s1', 0, 0, [0, 1], np.ones((100,)), 250, extent=1)
    s2 = PatchSource('s2', 1, 0, [0, 1], np.ones((100,)), 250, extent=1)
    s3 = PatchSource('s3', 0, 1, [0, 1], np.ones((100,)), 250, extent=1)

    # s1 and s2 are the same vertex, should be summed
    stc1 = _combine_sources_into_stc([s1, s2], src)
    assert stc1.data.shape[0] == 4, 'Expected 1 active vertices in stc'
    assert np.all(stc1.data == 1), 'Expected source activity not to be summed'

    # s1 and s3 are different vertices, should be concatenated
    stc2 = _combine_sources_into_stc([s1, s3], src)
    assert stc2.data.shape[0] == 2, 'Expected 2 active vertices in stc'
    assert np.all(stc2.data == 2), 'Expected source activity to be summed'


def test_patch_source_with_extent():
    """Test that PatchSource properly handles 'extent' parameter."""

    src = prepare_source_space(
        types=['surf', 'surf'],
        vertices=[[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
    )

    # Mock the times array
    n_samples = 100
    sfreq = 250
    times = np.arange(n_samples) / sfreq

    # Number of sources
    n_sources = 2

    # Mock location and waveform
    location = [(0, 2), (1, 8)]  # (src_idx, vertno)
    waveform = np.array([np.ones(n_samples,), np.ones(n_samples,)])  # Two waveforms for two sources
    names = ["source_1", "source_2"]

    # Mock extents for each source
    extents = [3, None]  # First source has extent, second has no extent

    # Mock grow_labels return values (for the first source with extent)
    patch_vertno = [2, 3, 4]  # Vertices grown for the first source

    with patch('mne.grow_labels') as mock_grow_labels:

        # Mock grow_labels to return the desired vertices
        mock_grow_labels.return_value = [MagicMock(vertices=patch_vertno)]

        # Call the create method
        sources = PatchSource.create(
            src=src,
            times=times,
            n_sources=n_sources,
            location=location,
            waveform=waveform,
            names=names,
            extents=extents,
            random_state=None
        )

        # Check that two sources were created
        assert len(sources) == 2, "The number of created sources is incorrect"

        # Check the first source (with extent)
        source_1 = sources[0]
        assert source_1.name == "source_1", "First source name mismatch"
        assert source_1.src_idx == 0, "First source src_idx mismatch"
        assert source_1.vertno == 2, "First source vertno mismatch"
        assert source_1.patch_vertno == patch_vertno, "First source patch_vertno mismatch"
        assert source_1.extent == 3, "First source extent mismatch"

        # Check the second source (without extent)
        source_2 = sources[1]
        assert source_2.name == "source_2", "Second source name mismatch"
        assert source_2.src_idx == 1, "Second source src_idx mismatch"
        assert source_2.vertno == 8, "Second source vertno mismatch"
        assert source_2.patch_vertno == [], "Second source patch_vertno should be empty"
        assert source_2.extent is None, "Second source extent mismatch"

        # Verify that grow_labels was called once for the source with extent
        mock_grow_labels.assert_called_once_with('meegsim', 2, 3, 0, subjects_dir=None)

