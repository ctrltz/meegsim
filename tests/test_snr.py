import numpy as np
import mne
from unittest.mock import patch

import pytest

from meegsim.snr import (
    get_sensor_space_variance, amplitude_adjustment_factor, _adjust_snr
)
from meegsim.source_groups import PointSourceGroup, PatchSourceGroup

from utils.prepare import (
    prepare_source_space, prepare_forward, 
    prepare_point_source, prepare_patch_source
)


def prepare_stc(vertices, num_samples=500):
    # Fill in dummy data as a constant time series equal to the vertex number
    data = np.tile(vertices[0] + vertices[1], reps=(num_samples, 1)).T
    return mne.SourceEstimate(data, vertices, tmin=0, tstep=0.01)


def test_get_sensor_space_variance_no_filter():
    fwd = prepare_forward(2, 4)
    fwd['sol']['data'] = np.array([
        [0, 1, 0, -1], 
        [0, -1, 0, 1]
    ])
    vertices = [[0, 1], [0, 1]]
    stc = prepare_stc(vertices)
    
    # Vertices with vertno=1 in both hemispheres have constant activity (1)
    # Since the leadfield values are opposite for these vertices, the
    # activity should cancel out in sensor space
    expected_variance = 0.
    variance = get_sensor_space_variance(stc, fwd, filter=False)
    assert np.isclose(variance, expected_variance), \
        f"Expected variance {expected_variance}, but got {variance}"


def test_get_sensor_space_variance_no_filter_sel_vert():
    fwd = prepare_forward(5, 10)
    vertices = [[0], [0]]
    stc = prepare_stc(vertices)
    
    # Both vertices in the stc have corresponding zero time series
    expected_variance = 0
    variance = get_sensor_space_variance(stc, fwd, filter=False)
    assert np.isclose(variance, expected_variance), \
        f"Expected variance {expected_variance}, but got {variance}"


@patch('meegsim.snr.filtfilt', return_value=np.ones((4, 500)))
@patch('meegsim.snr.butter', return_value=(0, 0))
def test_get_sensor_space_variance_with_filter(butter_mock, filtfilt_mock):
    fwd = prepare_forward(5, 10)
    vertices = [[0, 1], [0, 1]]
    stc = prepare_stc(vertices)
    variance = get_sensor_space_variance(stc, fwd, fmin=8, fmax=12, filter=True)

    # Check that butter and filtfilt were called
    butter_mock.assert_called()
    filtfilt_mock.assert_called()

    # Check that fmin and fmax are set to default values by looking at
    # the normalized frequencies (second argument of scipy.signal.butter)
    butter_args = butter_mock.call_args
    sfreq = stc.sfreq
    expected_wmin = 8.0 / (0.5 * sfreq)
    expected_wmax = 12.0 / (0.5 * sfreq)
    actual_wmin, actual_wmax = butter_args.args[1]
    assert np.isclose(actual_wmin, expected_wmin), \
        f"Expected fmin to be {expected_wmin}, got {actual_wmin}"
    assert np.isclose(actual_wmax, expected_wmax), \
        f"Expected fmax to be {expected_wmax}, got {actual_wmax}"

    assert variance >= 0, "Variance should be non-negative"


@patch('meegsim.snr.filtfilt', return_value=np.ones((4, 500)))
@patch('meegsim.snr.butter', return_value=(0, 0))
def test_get_sensor_space_variance_with_filter_fmin_fmax(butter_mock, filtfilt_mock):
    fwd = prepare_forward(5, 10)
    vertices = [[0, 1], [0, 1]]
    stc = prepare_stc(vertices)
    get_sensor_space_variance(stc, fwd, filter=True, fmin=20., fmax=30.)

    # Check that butter and filtfilt were called
    butter_mock.assert_called()
    filtfilt_mock.assert_called()

    # Check that fmin and fmax are set to custom values by looking at
    # the normalized frequencies (second argument of scipy.signal.butter)
    butter_args = butter_mock.call_args
    sfreq = stc.sfreq
    expected_wmin = 20.0 / (0.5 * sfreq)
    expected_wmax = 30.0 / (0.5 * sfreq)
    actual_wmin, actual_wmax = butter_args.args[1]
    assert np.isclose(actual_wmin, expected_wmin), \
        f"Expected fmin to be {expected_wmin}, got {actual_wmin}"
    assert np.isclose(actual_wmax, expected_wmax), \
        f"Expected fmax to be {expected_wmax}, got {actual_wmax}"


def test_get_sensor_space_variance_no_fmin_fmax():
    fwd = prepare_forward(5, 10)
    vertices = [[0, 1], [0, 1]]
    stc = prepare_stc(vertices)

    # No filtering required - should pass
    get_sensor_space_variance(stc, fwd, filter=False)

    # No fmin
    with pytest.raises(ValueError, match="Frequency band limits are required"):
        get_sensor_space_variance(stc, fwd, fmax=12, filter=True)

    # No fmax
    with pytest.raises(ValueError, match="Frequency band limits are required"):
        get_sensor_space_variance(stc, fwd, fmin=8, filter=True)


@pytest.mark.parametrize("target_snr", np.logspace(-6, 6, 10))
def test_amplitude_adjustment_factor(target_snr):
    signal_var = 10.0
    noise_var = 5.0

    snr_current = np.divide(signal_var, noise_var)
    expected_result = np.sqrt(target_snr / snr_current)

    result = amplitude_adjustment_factor(signal_var, noise_var, target_snr=target_snr)
    assert np.isclose(result, expected_result), \
        f"Expected {expected_result}, but got {result}"


def test_amplitude_adjustment_zero_signal_var():
    signal_var = 0.0
    noise_var = 5.0

    with pytest.raises(ValueError, match="initial SNR appear to be zero"):
        amplitude_adjustment_factor(signal_var, noise_var, target_snr=1)


def test_amplitude_adjustment_zero_noise_var():
    signal_var = 10.0
    noise_var = 0.0

    with pytest.raises(ValueError, match="noise variance appears to be zero"):
        amplitude_adjustment_factor(signal_var, noise_var, target_snr=1)


@patch('meegsim.snr.amplitude_adjustment_factor', return_value=2.)
def test_adjust_snr_point(adjust_snr_mock):
    src = prepare_source_space(
        types=['surf', 'surf'],
        vertices=[[0, 1], [0, 1]]
    )
    fwd = prepare_forward(5, 4)

    # Define source groups
    # SNR should be adjusted for s1 but not s2
    source_groups = [
        PointSourceGroup(
            n_sources=1, 
            location=[(0, 0)], 
            waveform=np.ones((1, 100)), 
            snr=np.array([5.]),
            snr_params=dict(fmin=8, fmax=12),
            names=['s1']
        ),
        PointSourceGroup(
            n_sources=1, 
            location=[(1, 0)], 
            waveform=np.ones((1, 100)), 
            snr=None, 
            snr_params=dict(),
            names=['s2']
        ),
    ]
    sources = {
        's1': prepare_point_source(name='s1'),
        's2': prepare_point_source(name='s2')
    }
    noise_sources = {
        'n1': prepare_point_source(name='n1')
    }
    tstep = 0.01

    sources = _adjust_snr(src, fwd, tstep, sources, source_groups, noise_sources)

    # Check the SNR adjustment was performed only once
    adjust_snr_mock.assert_called_once()

    # Check that the amplitude of s1 but not s2 was adjusted
    assert np.all(sources['s1'].waveform == 2)
    assert np.all(sources['s2'].waveform == 1)


@patch('meegsim.snr.amplitude_adjustment_factor', return_value=2.)
def test_adjust_snr_patch(adjust_snr_mock):
    src = prepare_source_space(
        types=['surf', 'surf'],
        vertices=[[0, 1], [0, 1]]
    )
    fwd = prepare_forward(5, 4)

    # Define source groups
    source_groups = [
        PatchSourceGroup(
            n_sources=1, 
            location=[(0, [0, 1])], 
            waveform=np.ones((1, 100)), 
            snr=np.array([5.]), 
            snr_params=dict(fmin=8, fmax=12), 
            extents=None,
            names=['s1']
        ),
        PatchSourceGroup(
            n_sources=1, 
            location=[(1, [0, 1])], 
            waveform=np.ones((1, 100)), 
            snr=None, 
            snr_params=dict(), 
            extents=None,
            names=['s2']
        ),
    ]
    sources = {
        's1': prepare_patch_source(name='s1'),
        's2': prepare_patch_source(name='s2')
    }
    noise_sources = {
        'n1': prepare_point_source(name='n1')
    }
    tstep = 0.01

    sources = _adjust_snr(src, fwd, tstep, sources, source_groups, noise_sources)

    # Check the SNR adjustment was performed only once
    adjust_snr_mock.assert_called_once()

    # Check that the amplitude of s1 but not s2 was adjusted
    assert np.all(sources['s1'].waveform == 2)
    assert np.all(sources['s2'].waveform == 2)


def test_adjust_snr_no_noise_sources_raises():
    src = prepare_source_space(
        types=['surf', 'surf'],
        vertices=[[0, 1], [0, 1]]
    )
    fwd = prepare_forward(5, 4)

    # it's only important that the noise sources list is empty
    with pytest.raises(ValueError, match="No noise sources"):
        _adjust_snr(src, fwd, 0.01, [], [], [])
