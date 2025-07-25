import numpy as np
import mne
from unittest.mock import patch

import pytest

from meegsim.snr import (
    get_variance,
    get_sensor_space_variance,
    amplitude_adjustment_factor,
    _adjust_snr_local,
    _adjust_snr_global,
)
from meegsim.source_groups import PointSourceGroup, PatchSourceGroup

from utils.prepare import (
    prepare_source_space,
    prepare_forward,
    prepare_point_source,
    prepare_patch_source,
    prepare_sinusoid,
)


def prepare_stc(vertices, num_samples=500):
    # Fill in dummy data as a constant time series equal to the vertex number
    data = np.tile(vertices[0] + vertices[1], reps=(num_samples, 1)).T
    return mne.SourceEstimate(data, vertices, tmin=0, tstep=0.01)


def test_get_variance():
    sfreq = 100
    waveform = prepare_sinusoid(f=10, sfreq=sfreq, duration=30)

    # int_0^{2*pi}(sin^2(x)dx) / (2 * pi) = 1/2
    assert np.isclose(get_variance(waveform, sfreq), 0.5)
    # amplitude x 2 -> variance x 4
    assert np.isclose(get_variance(2 * waveform, sfreq), 2)


def test_get_variance_with_filter():
    sfreq = 100
    waveform = prepare_sinusoid(f=10, sfreq=sfreq, duration=30)

    # We intentionally filter out the oscillation, so the variance should
    # be close to zero
    assert get_variance(waveform, sfreq, 20, 30, filter=True) < 1e-4


def test_get_sensor_space_variance_no_filter():
    fwd = prepare_forward(2, 4)
    fwd["sol"]["data"] = np.array([[0, 1, 0, -1], [0, -1, 0, 1]])
    vertices = [[0, 1], [0, 1]]
    stc = prepare_stc(vertices)

    # Vertices with vertno=1 in both hemispheres have constant activity (1)
    # Since the leadfield values are opposite for these vertices, the
    # activity should cancel out in sensor space
    expected_variance = 0.0
    variance = get_sensor_space_variance(stc, fwd, filter=False)
    assert np.isclose(
        variance, expected_variance
    ), f"Expected variance {expected_variance}, but got {variance}"


def test_get_sensor_space_variance_no_filter_sel_vert():
    fwd = prepare_forward(5, 10)
    vertices = [[0], [0]]
    stc = prepare_stc(vertices)

    # Both vertices in the stc have corresponding zero time series
    expected_variance = 0
    variance = get_sensor_space_variance(stc, fwd, filter=False)
    assert np.isclose(
        variance, expected_variance
    ), f"Expected variance {expected_variance}, but got {variance}"


@patch("meegsim.snr.filtfilt", return_value=np.ones((4, 500)))
@patch("meegsim.snr.butter", return_value=(0, 0))
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
    assert np.isclose(
        actual_wmin, expected_wmin
    ), f"Expected fmin to be {expected_wmin}, got {actual_wmin}"
    assert np.isclose(
        actual_wmax, expected_wmax
    ), f"Expected fmax to be {expected_wmax}, got {actual_wmax}"

    assert variance >= 0, "Variance should be non-negative"


@patch("meegsim.snr.filtfilt", return_value=np.ones((4, 500)))
@patch("meegsim.snr.butter", return_value=(0, 0))
def test_get_sensor_space_variance_with_filter_fmin_fmax(butter_mock, filtfilt_mock):
    fwd = prepare_forward(5, 10)
    vertices = [[0, 1], [0, 1]]
    stc = prepare_stc(vertices)
    get_sensor_space_variance(stc, fwd, filter=True, fmin=20.0, fmax=30.0)

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
    assert np.isclose(
        actual_wmin, expected_wmin
    ), f"Expected fmin to be {expected_wmin}, got {actual_wmin}"
    assert np.isclose(
        actual_wmax, expected_wmax
    ), f"Expected fmax to be {expected_wmax}, got {actual_wmax}"


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
    assert np.isclose(
        result, expected_result
    ), f"Expected {expected_result}, but got {result}"


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


# ====================
# _adjust_snr_local
# ====================


@patch("meegsim.snr.amplitude_adjustment_factor", return_value=2.0)
def test_adjust_snr_local_point(adjust_snr_mock):
    src = prepare_source_space(types=["surf", "surf"], vertices=[[0, 1], [0, 1]])
    fwd = prepare_forward(5, 4)

    # Define source groups
    # SNR should be adjusted for s1 but not s2
    source_groups = [
        PointSourceGroup(
            n_sources=1,
            location=[(0, 0)],
            waveform=np.ones((1, 100)),
            snr=np.array([5.0]),
            snr_params=dict(fmin=8, fmax=12),
            std=1,
            names=["s1"],
        ),
        PointSourceGroup(
            n_sources=1,
            location=[(1, 0)],
            waveform=np.ones((1, 100)),
            snr=None,
            snr_params=dict(),
            std=1,
            names=["s2"],
        ),
    ]
    sources = {
        "s1": prepare_point_source(name="s1"),
        "s2": prepare_point_source(name="s2"),
    }
    noise_sources = {"n1": prepare_point_source(name="n1")}
    tstep = 0.01

    _adjust_snr_local(src, fwd, tstep, sources, source_groups, noise_sources)

    # Check the SNR adjustment was performed only once
    adjust_snr_mock.assert_called_once()

    # Check that the amplitude of s1 but not s2 was adjusted
    assert np.all(sources["s1"].waveform == 2)
    assert np.all(sources["s2"].waveform == 1)


@patch("meegsim.snr.amplitude_adjustment_factor", return_value=2.0)
def test_adjust_snr_local_patch(adjust_snr_mock):
    src = prepare_source_space(types=["surf", "surf"], vertices=[[0, 1], [0, 1]])
    fwd = prepare_forward(5, 4)

    # Define source groups
    source_groups = [
        PatchSourceGroup(
            n_sources=1,
            location=[(0, [0, 1])],
            waveform=np.ones((1, 100)),
            snr=np.array([5.0]),
            snr_params=dict(fmin=8, fmax=12),
            std=1,
            extents=None,
            subject=None,
            subjects_dir=None,
            names=["s1"],
        ),
        PatchSourceGroup(
            n_sources=1,
            location=[(1, [0, 1])],
            waveform=np.ones((1, 100)),
            snr=None,
            snr_params=dict(),
            std=1,
            extents=None,
            subject=None,
            subjects_dir=None,
            names=["s2"],
        ),
    ]
    sources = {
        "s1": prepare_patch_source(name="s1"),
        "s2": prepare_patch_source(name="s2"),
    }
    noise_sources = {"n1": prepare_point_source(name="n1")}
    tstep = 0.01

    _adjust_snr_local(src, fwd, tstep, sources, source_groups, noise_sources)

    # Check the SNR adjustment was performed only once
    adjust_snr_mock.assert_called_once()

    # Check that the amplitude of s1 but not s2 was adjusted
    assert np.all(sources["s1"].waveform == 2)
    assert np.all(sources["s2"].waveform == 1)


def test_adjust_snr_local_no_noise_sources_raises():
    src = prepare_source_space(types=["surf", "surf"], vertices=[[0, 1], [0, 1]])
    fwd = prepare_forward(5, 4)

    # it's only important that the noise sources list is empty
    with pytest.raises(ValueError, match="No noise sources"):
        _adjust_snr_local(src, fwd, 0.01, [], [], [])


# ====================
# _adjust_snr_global
# ====================


@patch("meegsim.snr.amplitude_adjustment_factor", return_value=2.0)
def test_adjust_snr_global_point(adjust_snr_mock):
    src = prepare_source_space(types=["surf", "surf"], vertices=[[0, 1], [0, 1]])
    fwd = prepare_forward(5, 4)

    # Define sources
    # SNR should be adjusted for both s1 and s2 by the same factor
    sources = {
        "s1": prepare_point_source(name="s1"),
        "s2": prepare_point_source(name="s2"),
    }
    noise_sources = {"n1": prepare_point_source(name="n1")}
    tstep = 0.01

    _adjust_snr_global(
        src,
        fwd,
        snr_global=5,
        snr_params=dict(fmin=8, fmax=12),
        tstep=tstep,
        sources=sources,
        noise_sources=noise_sources,
    )

    # Check the SNR adjustment was performed only once
    adjust_snr_mock.assert_called_once()

    # Check that the amplitudes of s1 and s2 were adjusted equally
    assert np.all(sources["s1"].waveform == 2)
    assert np.all(sources["s2"].waveform == 2)


@patch("meegsim.snr.amplitude_adjustment_factor", return_value=2.0)
def test_adjust_snr_global_patch(adjust_snr_mock):
    src = prepare_source_space(types=["surf", "surf"], vertices=[[0, 1], [0, 1]])
    fwd = prepare_forward(5, 4)

    # Define sources
    # SNR should be adjusted for both s1 and s2 by the same factor
    sources = {
        "s1": prepare_patch_source(name="s1"),
        "s2": prepare_patch_source(name="s2"),
    }
    noise_sources = {"n1": prepare_point_source(name="n1")}
    tstep = 0.01

    _adjust_snr_global(
        src,
        fwd,
        snr_global=5,
        snr_params=dict(fmin=8, fmax=12),
        tstep=tstep,
        sources=sources,
        noise_sources=noise_sources,
    )

    # Check the SNR adjustment was performed only once
    adjust_snr_mock.assert_called_once()

    # Check that the amplitudes of s1 and s2 were adjusted equally
    assert np.all(sources["s1"].waveform == 2)
    assert np.all(sources["s2"].waveform == 2)


def test_adjust_snr_global_no_sources_warns():
    src = prepare_source_space(types=["surf", "surf"], vertices=[[0, 1], [0, 1]])
    fwd = prepare_forward(5, 4)

    # it's only important that the noise sources list is empty
    with pytest.warns(UserWarning, match="No point/patch sources"):
        _adjust_snr_global(src, fwd, 5, dict(fmin=8, fmax=12), 0.01, dict(), [])


def test_adjust_snr_global_no_noise_sources_raises():
    src = prepare_source_space(types=["surf", "surf"], vertices=[[0, 1], [0, 1]])
    fwd = prepare_forward(5, 4)

    sources = {
        "s1": prepare_patch_source(name="s1"),
        "s2": prepare_patch_source(name="s2"),
    }

    # it's only important that the noise sources list is empty
    with pytest.raises(ValueError, match="No noise sources"):
        _adjust_snr_global(src, fwd, 5, dict(fmin=8, fmax=12), 0.01, sources, [])
