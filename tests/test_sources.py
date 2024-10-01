import numpy as np
import pytest

from meegsim.sources import _BaseSource, PointSource, _combine_sources_into_stc

from utils.prepare import prepare_source_space


def test_basesource_is_abstract():
    waveform = np.ones((100,))
    s = _BaseSource(waveform)
    with pytest.raises(NotImplementedError, match="in the subclass"):
        s.data()


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
    s = PointSource('mysource', src_idx, vertno, np.array([]), hemi=hemi)

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
    s = PointSource('mysource', src_idx, vertno, waveform)
    stc = s.to_stc(src, tstep=0.01)

    assert stc.data.shape[0] == 1, \
        f"Expected one active vertex in stc, got {stc.data.shape[0]}"
    assert vertno in stc.vertices[src_idx], \
        f"Expected the vertex to be put in src {src_idx}, but it is not there"
    assert np.allclose(stc.data, waveform), \
        f"The source waveform should not change during conversion to stc"
    

@pytest.mark.parametrize("tstep", [0.01, 0.02, 0.05])
def test_pointsource_to_stc_tstep(tstep):
    waveform = np.ones((100,))
    src = prepare_source_space(
        types=['surf', 'surf'],
        vertices=[[0, 1], [0, 1]]
    )
    s = PointSource('mysource', 0, 0, waveform)
    stc = s.to_stc(src, tstep)

    expected_sfreq = 1.0 / tstep
    assert stc.sfreq == expected_sfreq, \
        f"Expected stc.sfreq to be {expected_sfreq}, got {stc.sfreq}"
    

def test_pointsource_to_stc_subject():
    waveform = np.ones((100,))
    src = prepare_source_space(
        types=['surf', 'surf'],
        vertices=[[0, 1], [0, 1]]
    )
    s = PointSource('mysource', 0, 0, waveform)
    stc = s.to_stc(src, tstep=0.01)

    assert stc.subject == 'meegsim', \
        f"Expected stc.subject to be derived from src, got {stc.subject}"
    
    stc = s.to_stc(src, subject='mysubject', tstep=0.01)

    assert stc.subject == 'mysubject', \
        f"Expected stc.subject to be mysubject, got {stc.subject}"
    

def test_pointsource_to_stc_bad_src_raises():
    waveform = np.ones((100,))
    src = prepare_source_space(
        types=['surf', 'surf'],
        vertices=[[0, 1], [0, 1]]
    )

    # src[2] is out of range
    s = PointSource('mysource', 2, 0, waveform)
    with pytest.raises(ValueError, match="not present in the provided src"):
        s.to_stc(src, subject='mysubject', tstep=0.01)


def test_pointsource_to_stc_bad_vertno_raises():
    waveform = np.ones((100,))
    src = prepare_source_space(
        types=['surf', 'surf'],
        vertices=[[0, 1], [0, 1]]
    )

    # vertex 2 is not in src[0]
    s = PointSource('mysource', 0, 2, waveform)
    with pytest.raises(ValueError, match="does not contain the vertex"):
        s.to_stc(src, subject='mysubject', tstep=0.01)


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

    s1 = PointSource('s1', 0, 0, np.ones((100,)))
    s2 = PointSource('s2', 0, 0, np.ones((100,)))
    s3 = PointSource('s3', 0, 1, np.ones((100,)))

    # s1 and s2 are the same vertex, should be summed
    stc1 = _combine_sources_into_stc([s1, s2], src, 0.01)
    assert stc1.data.shape[0] == 1, 'Expected 1 active vertices in stc'
    assert np.all(stc1.data == 2), 'Expected source activity to be summed'

    # s1 and s3 are different vertices, should be concatenated
    stc2 = _combine_sources_into_stc([s1, s3], src, 0.01)
    assert stc2.data.shape[0] == 2, 'Expected 2 active vertices in stc'
    assert np.all(stc2.data == 1), 'Expected source activity not to be summed'
