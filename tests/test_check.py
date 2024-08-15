import numpy as np
import pytest

from functools import partial
from meegsim._check import (
    check_callable, check_vertices_list_of_tuples, check_vertices_in_src,
    check_location, check_waveform
)

from utils.prepare import prepare_source_space


def test_check_callable():
    def good_callable(*args, **kwargs):
        args_dict = {f'arg{i}': arg for i, arg in enumerate(args)}
        args_dict.update(kwargs)

        return args_dict
    
    result = check_callable('good', good_callable, 'arg', kwarg='kwarg')
    
    # Check that all arguments were passed correctly
    assert result['arg0'] == 'arg'
    assert result['kwarg'] == 'kwarg'

    # Always calling with random_state of 0
    assert result['random_state'] == 0


def test_check_callable_raises():
    def bad_callable(*args, **kwargs):
        raise ValueError("original error message")
    
    with pytest.raises(ValueError, match="original error message"):
        check_callable('bad', bad_callable, 'arg', kwarg='kwarg')


def test_check_vertices_list_of_tuples():
    check_vertices_list_of_tuples([(0, 0), (0, 1), (1, 1)])


def test_check_vertices_list_of_tuples_raises():
    with pytest.raises(ValueError, match="vertices to be a list or a tuple"):
        check_vertices_list_of_tuples('aaa')

    with pytest.raises(ValueError, match="to be a list or a tuple, does not hold for element 1"):
        check_vertices_list_of_tuples([(0, 1), 1])

    with pytest.raises(ValueError, match="contain 2 values, does not hold for element \(1, 2, 3\)"):
        check_vertices_list_of_tuples([(0, 1), (0, 2), (1, 2, 3)])


def test_check_vertices_in_src():
    src = prepare_source_space(
        types=['surf', 'surf'],
        vertices=[[0, 1], [0, 1]]
    )

    # should pass
    check_vertices_in_src([(0, 0), (0, 1), (1, 0), (1, 1)], src)


def test_check_vertices_in_src_raises():
    src = prepare_source_space(
        types=['surf', 'surf'],
        vertices=[[0, 1], [0, 1]]
    )

    with pytest.raises(ValueError, match='Vertex \(2, 0\) belongs to'):
        check_vertices_in_src([(0, 0), (1, 0), (2, 0)], src)

    with pytest.raises(ValueError, match='Vertex \(0, 2\) is not present'):
        check_vertices_in_src([(0, 0), (0, 1), (0, 2)], src)


def test_check_location_using_arrays():
    location = [(0, 0), (1, 1)]
    src = prepare_source_space(
        types=['surf', 'surf'],
        vertices=[[0, 1], [0, 1]]
    )
    checked, n_vertices = check_location(
        location, 
        dict(pick=1),     # should be ignored
        src
    )

    assert checked == location, \
        "The provided value for location was changed"
    assert n_vertices == 2, "Wrong number of vertices"
    

def test_check_location_using_callables():
    def location_fun(src, pick=0, random_state=None):
        # pick the desired vertex in each source space
        return [
            (src_idx, s['vertno'][pick]) 
            for src_idx, s in enumerate(src)
        ]
    
    src = prepare_source_space(
        types=['surf', 'surf'],
        vertices=[[0, 1], [0, 1]]
    )
    checked, n_vertices = check_location(
        location_fun, 
        dict(pick=1),
        src
    )

    assert isinstance(checked, partial), \
        "Expected the location function to be converted to a partial object"
    assert checked.keywords['pick'] == 1, \
        "The provided value of the keyword argument was changed"
    assert n_vertices == 2, "Wrong number of vertices"


def test_check_waveform_with_arrays():
    waveform = np.ones((2, 100))
    checked = check_waveform(
        waveform, 
        dict(value=1),     # should be ignored
        n_sources=2
    )

    assert np.array_equal(checked, waveform), \
        "The provided waveform was changed"
    

def test_check_waveform_array_bad_shape_raises():
    waveform = np.ones((5, 100))
    with pytest.raises(ValueError, match="number of sources"):
        check_waveform(
            waveform, 
            dict(),     
            n_sources=2        # expected 2 sources but will get 5
        )


def test_check_waveform_using_callables():
    def waveform_fun(n_series, times, value=0, random_state=None):
        # return constant time series equal to the provided value
        return np.ones((n_series, len(times))) * value

    checked = check_waveform(
        waveform_fun, 
        dict(value=1),
        n_sources=2
    )

    assert isinstance(checked, partial), \
        "Expected the waveform function to be converted to a partial object"
    assert checked.keywords['value'] == 1, \
        "The provided value of the keyword argument was changed"


def test_check_waveform_callable_bad_shape_raises():
    def waveform_fun(n_series, times, random_state=None):
        # return constant time series, ignore the requested size
        return np.ones((5, 100))
    
    # raises an error since 1000 samples are requested when testing the waveform function
    with pytest.raises(ValueError, match="number of samples"):
        check_waveform(
            waveform_fun, 
            dict(),     
            n_sources=5        
        )

    with pytest.raises(ValueError, match="number of sources"):
        check_waveform(
            waveform_fun, 
            dict(),     
            n_sources=2       # expected 2 sources, will get 5 
        )