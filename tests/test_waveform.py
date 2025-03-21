import numpy as np
import pytest

from mock import patch
from scipy.signal import welch

from meegsim.utils import get_sfreq
from meegsim.waveform import white_noise, narrowband_oscillation, one_over_f_noise


def prepare_times(sfreq, duration):
    n_times = sfreq * duration
    times = np.arange(n_times) / sfreq
    return n_times, times


@pytest.mark.parametrize(
    "waveform,waveform_params",
    [
        (narrowband_oscillation, dict(fmin=8, fmax=12)),
        (narrowband_oscillation, dict(fmin=16, fmax=24)),
        (one_over_f_noise, dict(slope=1)),
        (one_over_f_noise, dict(slope=2)),
        (white_noise, dict()),
    ],
)
def test_waveforms_random_state(waveform, waveform_params):
    """
    Test that all waveforms support random state.
    """
    n_series = 10
    _, times = prepare_times(sfreq=250, duration=30)

    # Different time series are generated by default
    data1 = waveform(n_series, times, **waveform_params)
    data2 = waveform(n_series, times, **waveform_params)
    assert not np.allclose(data1, data2)

    # The results are reproducible when random_state is set
    random_state = 1234567890
    data1 = waveform(n_series, times, random_state=random_state, **waveform_params)
    data2 = waveform(n_series, times, random_state=random_state, **waveform_params)
    assert np.allclose(data1, data2)


@pytest.mark.parametrize(
    "waveform,waveform_params",
    [
        (narrowband_oscillation, dict(fmin=8, fmax=12)),
        (narrowband_oscillation, dict(fmin=16, fmax=24)),
        (one_over_f_noise, dict(slope=1)),
        (one_over_f_noise, dict(slope=2)),
        (white_noise, dict()),
    ],
)
def test_waveforms_shape(waveform, waveform_params):
    """
    Test that the result of all waveform functions has correct shape.
    """
    n_series = 10
    n_times, times = prepare_times(sfreq=250, duration=30)

    data = waveform(n_series, times, **waveform_params)
    assert data.shape == (n_series, n_times)


@pytest.mark.parametrize(
    "fmin, fmax",
    [
        (4.0, 7.0),
        (8.0, 12.0),
        (20.0, 30.0),
        (15.0, 35.0),
    ],
)
def test_narrowband_oscillation_fmin_fmax(fmin, fmax):
    """
    Test that frequencies within the specified band have higher power
    than the rest of the spectra.
    """
    n_series = 10
    n_times, times = prepare_times(sfreq=250, duration=30)

    data = narrowband_oscillation(n_series, times, fmin=fmin, fmax=fmax)

    # Calculate power spectral density
    fs = get_sfreq(times)
    freqs, power = welch(data, fs=fs, nfft=fs, nperseg=fs, axis=1)

    # Sort frequencies by power
    sorted_freqs = freqs[np.argsort(power.mean(axis=0))[::-1]]

    # Check if frequencies within the band are among the most powerful
    band_fmin_fmax = (freqs >= fmin) & (freqs <= fmax)
    band_freqs = sorted_freqs[: np.sum(band_fmin_fmax)]
    assert len(band_freqs) > 0, "No frequencies found in the specified band."
    assert np.all(
        (band_freqs >= fmin) & (band_freqs <= fmax)
    ), "Not all powerful frequencies are in the specified band."
    assert data.shape == (n_series, n_times), "Shape mismatch"


# return dummy values for the function to run
# import the functions from our module to resolve 'from ... import ...' definition
# more about: https://nedbatchelder.com/blog/201908/why_your_mock_doesnt_work.html
@patch("meegsim.waveform.filtfilt", return_value=np.ones((1, 100)))
@patch("meegsim.waveform.butter", return_value=(0, 0))
def test_narrowband_oscillation_order(butter_mock, filtfilt_mock):
    _, times = prepare_times(sfreq=250, duration=30)

    # order is set to 2 by default
    narrowband_oscillation(n_series=10, times=times, fmin=8, fmax=12)
    butter_mock.assert_called()
    assert butter_mock.call_args.kwargs["N"] == 2

    # custom slope value also should work
    narrowband_oscillation(n_series=10, times=times, fmin=8, fmax=12, order=4)
    assert butter_mock.call_args.kwargs["N"] == 4


# return a dummy value for normalize_power to work
@patch("colorednoise.powerlaw_psd_gaussian", return_value=np.ones((1, 100)))
def test_one_over_f_noise_slope(noise_mock):
    _, times = prepare_times(sfreq=250, duration=30)

    # slope is set to 1 by default
    one_over_f_noise(n_series=10, times=times)
    assert 1 in noise_mock.call_args.args

    # custom slope value also should work
    one_over_f_noise(n_series=10, times=times, slope=1.5)
    assert 1.5 in noise_mock.call_args.args
