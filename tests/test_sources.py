# TODO (this PR):
#  - tests for PointSource
#  - tests for create_point_sources?

import numpy as np
import pytest

from meegsim.sources import BaseSource, PointSource

from utils.prepare import prepare_source_space


def test_basesource_is_abstract():
    waveform = np.ones((100,))
    s = BaseSource(waveform)
    with pytest.raises(NotImplementedError, match="in the subclass"):
        s.to_stc()


@pytest.mark.parametrize(
    "src_idx,vertno,hemi", [
        (0, 123, None),
        (0, 234, 'lh'),
        (1, 345, None),
        (1, 456, 'rh'),
    ]
)
def test_pointsource_repr(src_idx, vertno, hemi):
    s = PointSource(src_idx, vertno, np.array([]), hemi)

    if hemi is None:
        assert f'src[{src_idx}]' in repr(s)
    else:
        assert hemi in repr(s)

    assert str(vertno) in repr(s)


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
    s = PointSource(src_idx, vertno, waveform)
    stc = s.to_stc(src, sfreq=100)

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
    s = PointSource(0, 0, waveform)
    stc = s.to_stc(src, sfreq=sfreq)

    assert stc.sfreq == sfreq, \
        f"Expected stc.sfreq to be {sfreq}, got {stc.sfreq}"
    

def test_pointsource_to_stc_subject():
    waveform = np.ones((100,))
    src = prepare_source_space(
        types=['surf', 'surf'],
        vertices=[[0, 1], [0, 1]]
    )
    s = PointSource(0, 0, waveform)
    stc = s.to_stc(src, sfreq=250, subject='mysubject')

    assert stc.subject == 'mysubject', \
        f"Expected stc.sfreq to be mysubject, got {stc.subject}"
    

def test_pointsource_to_stc_bad_src_raises():
    waveform = np.ones((100,))
    src = prepare_source_space(
        types=['surf', 'surf'],
        vertices=[[0, 1], [0, 1]]
    )

    # src[2] is out of range
    s = PointSource(2, 0, waveform)
    with pytest.raises(ValueError, match="not present in the provided src"):
        s.to_stc(src, sfreq=250, subject='mysubject')


def test_pointsource_to_stc_bad_vertno_raises():
    waveform = np.ones((100,))
    src = prepare_source_space(
        types=['surf', 'surf'],
        vertices=[[0, 1], [0, 1]]
    )

    # vertex 2 is not in src[0]
    s = PointSource(0, 2, waveform)
    with pytest.raises(ValueError, match="does not contain the vertex"):
        s.to_stc(src, sfreq=250, subject='mysubject')
