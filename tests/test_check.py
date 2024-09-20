import numpy as np
import pytest

from functools import partial
from meegsim._check import (
    check_callable, check_vertices_list_of_tuples, check_vertices_in_src,
    check_location, check_waveform, check_names, check_snr, check_snr_params,
    check_if_source_exists, check_coupling, check_coupling_params
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


def test_check_snr_is_none_passes():
    snr = check_snr(None, 5)
    assert snr is None


@pytest.mark.parametrize("n_sources", [1, 5, 10])
def test_check_snr_float_passes(n_sources):
    snr = check_snr(1., n_sources)
    assert snr.size == n_sources
    assert np.all(snr == 1.)


def test_check_snr_array_valid_shape_passes():
    initial = [1, 2, 3, 4, 5]
    snr = check_snr(initial, 5)
    assert np.array_equal(snr, initial)


def test_check_snr_array_invalid_shape_raises():
    initial = [1, 2, 3, 4, 5]
    with pytest.raises(ValueError, match="of the 3 sources, got 5"):
        check_snr(initial, 3)


def test_check_snr_negative_snr_raises():
    with pytest.raises(ValueError, match="Each SNR value should be positive"):
        check_snr([-1, 0, 1], 3)


def test_check_snr_params_snr_is_none():
    snr_params = check_snr_params(dict(), None)
    assert not snr_params


def test_check_snr_params_no_fmin():
    with pytest.raises(ValueError, match="Please add fmin and fmax"):
        check_snr_params(dict(fmax=12.), snr=1.)


def test_check_snr_params_no_fmax():
    with pytest.raises(ValueError, match="Please add fmin and fmax"):
        check_snr_params(dict(fmin=8.), snr=1.)


def test_check_snr_params_negative_fmin_fmax():
    with pytest.raises(ValueError, match="Frequency limits should be positive"):
        check_snr_params(dict(fmin=-8., fmax=12.), snr=1.)
        
    with pytest.raises(ValueError, match="Frequency limits should be positive"):
        check_snr_params(dict(fmin=8., fmax=-12.), snr=1.)


def test_check_names_should_pass():
    initial = ['m1-lh', 'm1-rh', 's1-lh', 's1-rh']
    check_names(initial, 4, ['v1-lh', 'v1-rh'])


def test_check_names_wrong_number():
    with pytest.raises(ValueError, match='does not match the number of defined'):
        check_names(['a', 'b', 'c'], 5, [])


def test_check_names_non_unique():
    with pytest.raises(ValueError, match='should be unique'):
        check_names(['a', 'a', 'b'], 3, [])


def test_check_names_wrong_type():
    with pytest.raises(ValueError, match='to be strings, got int: 1'):
        check_names(['a', 'b', 1], 3, [])
    with pytest.raises(ValueError, match='to be strings, got list'):
        check_names(['a', 'b', ['c', 'd']], 3, [])


def test_check_names_empty():
    with pytest.raises(ValueError, match='should not be empty'):
        check_names(['a', 'b', ''], 3, [])


def test_check_names_auto():
    with pytest.raises(ValueError, match='Name auto-s1 should not start with auto'):
        check_names(['a', 'b', 'auto-s1'], 3, [])


def test_check_names_already_exists():
    with pytest.raises(ValueError, match='Name s1 is already taken'):
        check_names(['a', 'b', 's1'], 3, ['s1'])


def test_check_if_source_exists_should_pass():
    existing = ['aaa', 'bbb']
    check_if_source_exists('aaa', existing)


def test_check_if_source_exists_should_raise():
    existing = ['aaa', 'bbb']
    with pytest.raises(ValueError, match='Source ccc'):
        check_if_source_exists('ccc', existing)


def test_check_coupling_params_should_pass():
    edge = ('0', '1')
    method = 'ppc_von_mises'
    coupling_params = dict(
        phase_lag=0,
        kappa=1,
        fmin=8,
        fmax=12
    )

    check_coupling_params(method, coupling_params, edge)


def test_check_coupling_params_should_raise():
    edge = ('0', '1')
    method = 'ppc_von_mises'
    coupling_params = dict(
        phase_lag=0,
        kappa=1,
        fmin=8,
    )
    expected_message = 'The fmax parameter is required for the ppc_von_mises method'

    with pytest.raises(ValueError, match=expected_message):
        check_coupling_params(method, coupling_params, edge)


def test_check_coupling_should_pass():
    sources = ['a', 'b', 'c', 'd']
    existing = {}
    coupling_params = {'kappa': 1, 'phase_lag': 0}
    common = {'method': 'ppc_von_mises', 'fmin': 8, 'fmax': 12}

    params = check_coupling(('a', 'b'), coupling_params, common, sources, existing)

    # Check that common params was added to the dictionary
    assert 'method' in params
    assert 'fmin' in params
    assert 'fmax' in params


def test_check_coupling_bad_source_name():
    sources = ['a', 'b', 'c', 'd']

    # Bad target node
    with pytest.raises(ValueError, match="Source e was not defined yet"):
        check_coupling(('a', 'e'), {}, {}, sources, {})

    # Bad source node
    with pytest.raises(ValueError, match="Source f was not defined yet"):
        check_coupling(('f', 'b'), {}, {}, sources, {})


def test_check_coupling_edge_already_exists():
    sources = ['a', 'b', 'c', 'd']
    existing = {
        ('a', 'b'): dict(),
    }

    with pytest.raises(ValueError, match="multiple definitions are not allowed"):
        check_coupling(('a', 'b'), {}, {}, sources, existing)


def test_check_coupling_no_method_defined():
    sources = ['a', 'b', 'c', 'd']
    existing = {}
    coupling_params = {'kappa': 1, 'phase_lag': 0}

    with pytest.raises(ValueError, match="method was not defined"):
        check_coupling(('a', 'b'), coupling_params, {}, sources, existing)


def test_check_coupling_bad_params():
    sources = ['a', 'b', 'c', 'd']
    existing = {}
    coupling_params = {'kappa': 1, 'phase_lag': 0, 'fmin': 8}
    common = {'method': 'ppc_von_mises'}

    with pytest.raises(ValueError, match="The fmax parameter is required"):
        check_coupling(('a', 'b'), coupling_params, common, sources, existing)
