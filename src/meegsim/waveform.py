"""
All waveform functions should accept the following arguments:
    * the shape of time series to generate (number of time series, time samples)
    * all waveform-specific arguments should be keyword-only
    * ideally, random_state (to allow reproducibility, still need to test how it would work)

This can be achieved with the following template:

def waveform_fn(n_series, times, *, kwarg1='aaa', kwarg2='bbb'):
    pass
    
Waveforms currently in mind:
    * narrowband_oscillation (a.k.a. filtered white noise)
    * one_over_f_noise
    * white_noise (for sensor space noise)

Waveforms that are not urgent to have but could in principle be useful:
    * non-sinusoidal stuff (harmonics, peak-trough asymmetry)
"""
import warnings

import numpy as np

from .utils import normalize_power


def narrowband_oscillation(n_series, times, fs, *, fmin=None, fmax=None, order=2, random_state=None):
    #
    # Ideas for tests
    # Test 2 (order)
    #  - we could check that we pass correct value to filtfilt (requires mocking)
    #  - check the shape of the resulting array
    """
    Generate time series in a requested frequency band by filtering white noise

    Parameters
    ----------
    n_series: int
        Number of time series to generate.

    times: ndarray
        Array of time points (each one represents time in seconds).

    fs: float
        Sampling frequency (in Hz).

    fmin: float, optional
        Lower cutoff frequency (in Hz). default = None.

    fmax: float, optional
        Upper cutoff frequency (in Hz). default = None.

    order: int, optional
        The order of the filter. default = 2.

    random_state: int or None, optional
        Seed for the random number generator. If None, it will be drawn
        automatically, and results will vary between function calls.

    Returns
    -------
    out: ndarray, shape (n_series, n_times)
        Generated filtered white noise.
    """

    from scipy.signal import butter, filtfilt

    if fmin is None:
        warnings.warn("fmin was None. Setting fmin to 8 Hz", UserWarning)
        fmin = 8.
    if fmax is None:
        warnings.warn("fmax was None. Setting fmax to 12 Hz", UserWarning)
        fmax = 12.

    if fmin >= fmax:
        raise ValueError("fmin must be smaller than fmax.")
    if fmin <= 0 or fmax <= 0:
        raise ValueError("filter frequencies must be greater than 0")

    if not isinstance(order, int) or order <= 0:
        raise ValueError("order must be a positive integer.")

    rng = np.random.default_rng(seed=random_state)
    data = rng.standard_normal(size=(n_series, times.size))
    b1, a1 = butter(N=order, Wn=np.array([fmin, fmax]) / fs * 2, btype='bandpass')
    data = filtfilt(b1, a1, data, axis=1)
    return normalize_power(data)


def one_over_f_noise(n_series, times, *, slope=1, random_state=None):
    """
    Generate time series of pink noise

    Parameters
    ----------
    n_series: int
        Number of time series to generate.

    times: ndarray
        Array of time points (each one represents time in seconds).

    slope: float, optional
        Exponent of the power-law spectrum. default = 1.

    random_state: int or None, optional
        Seed for the random number generator. If None, it will be drawn
        automatically, and results will vary between function calls.

    Returns
    -------
    out: ndarray, shape (n_series, n_times)
        Generated filtered white noise.
    """

    import colorednoise as cn

    data = cn.powerlaw_psd_gaussian(slope, size=(n_series, times.size), random_state=random_state)
    return normalize_power(data)
    

def white_noise(n_series, times, *, random_state=None):
    """
    Generate time series of white noise (e.g., to use for modeling
    measurement noise in sensor space data)

    Parameters
    ----------
    n_series: int
        Number of time series to generate.

    times: ndarray
        Array of time points (each one represents time in seconds).

    random_state: int
        Seed for the random number generator. If None, it will be drawn
        automatically, and results will vary between function calls.

    Returns
    -------
    out: ndarray, shape (n_series, n_times)
        Generated white noise.
    """

    rng = np.random.default_rng(seed=random_state)
    data = rng.standard_normal(size=(n_series, times.size))
    return normalize_power(data)