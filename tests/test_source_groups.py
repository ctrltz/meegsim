import numpy as np
import pytest

from functools import partial
from mock import patch
from meegsim.source_groups import (
    _BaseSourceGroup, PointSourceGroup, generate_names
)

from utils.prepare import prepare_source_space


def check_all_mocks(sg, mocks):
    for field, mock in mocks.items():
        mock.assert_called()
        assert getattr(sg, field) == f'mock {field}', \
            f"Value of {field} was stored incorrectly"


@pytest.mark.parametrize(
    "group,n_sources", 
    [
        ('group1', 3),
        ('group2', 6)
    ]
)
def test_generate_names(group, n_sources):
    names = generate_names(group, n_sources)
    assert len(names) == n_sources
    assert all([f'auto-{group}-s' in name for name in names])


def test_basesourcegroup_is_abstract():
    with pytest.raises(NotImplementedError, match="implemented in a subclass"):
        _BaseSourceGroup().simulate()


def test_pointsourcegroup_repr_no_callables():
    point_sg = PointSourceGroup(4, [(0, 0)], np.array([0]), None, dict(), [])
    assert '4 sources' in repr(point_sg)
    assert 'location=list' in repr(point_sg)
    assert 'waveform=array' in repr(point_sg)


def test_pointsourcegroup_repr_with_callables():
    def my_location(x):
        return x
    
    def my_waveform(x):
        return x

    point_sg = PointSourceGroup(
        4, 
        partial(my_location, x=1), 
        partial(my_waveform, x=1), 
        None,
        dict(),
        []
    )
    assert '4 sources' in repr(point_sg)
    assert 'location=my_location' in repr(point_sg)
    assert 'waveform=my_waveform' in repr(point_sg)


@patch('meegsim.source_groups.check_waveform', return_value='mock waveform')
@patch('meegsim.source_groups.check_location', return_value=('mock location', 0))
def test_pointsourcegroup_create_using_arrays(location_mock, waveform_mock):
    location = [(0, 0), (1, 1)]
    waveform = np.array((2, 100))
    src = prepare_source_space(
        types=['surf', 'surf'],
        vertices=[[0, 1], [0, 1]]
    )
    point_sg = PointSourceGroup.create(
        src,
        location,
        waveform,
        snr=None,
        location_params=dict(pick=1),
        waveform_params=dict(value=1),
        snr_params=dict(),
        names=None,
        group=None,
        existing=[]
    )

    # Check the type of the returned object
    assert isinstance(point_sg, PointSourceGroup), "Wrong object type"

    # Check that the data is stored correctly
    mocks = {
        'location': location_mock,
        'waveform': waveform_mock
    }
    check_all_mocks(point_sg, mocks)


@patch('meegsim.source_groups.check_waveform', return_value='mock waveform')
@patch('meegsim.source_groups.check_location', return_value=('mock location', 0))
def test_pointsourcegroup_create_using_callables(location_mock, waveform_mock):
    def location_fun(src, pick=0, random_state=None):
        # pick the desired vertex in each source space
        return [
            (src_idx, s['vertno'][pick]) 
            for src_idx, s in enumerate(src)
        ]
    
    def waveform_fun(n_series, times, value=0, random_state=None):
        # return constant time series equal to the provided value
        return np.ones((n_series, len(times))) * value
    
    src = prepare_source_space(
        types=['surf', 'surf'],
        vertices=[[0, 1], [0, 1]]
    )
    point_sg = PointSourceGroup.create(
        src,
        location_fun,
        waveform_fun,
        snr=None,
        location_params=dict(pick=1),
        waveform_params=dict(value=1),
        snr_params=dict(),
        names=None,
        group=None,
        existing=[]
    )

    # Check the type of the returned object
    assert isinstance(point_sg, PointSourceGroup), "Wrong object type"

    # Check that the data is stored correctly
    mocks = {
        'location': location_mock,
        'waveform': waveform_mock
    }
    check_all_mocks(point_sg, mocks)