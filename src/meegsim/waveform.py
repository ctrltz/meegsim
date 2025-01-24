"""
Template waveforms: narrowband oscillation, white and 1/f noise
"""

import colorednoise as cn
import numpy as np
import warnings

from scipy.signal import butter, filtfilt

from .utils import normalize_variance, get_sfreq


def narrowband_oscillation(
    n_series, times, *, fmin=None, fmax=None, order=2, random_state=None
):
    """
    Generate oscillatory-like time series by filtering white noise
    in a requested frequency band.

    Parameters
    ----------
    n_series : int
        The number of time series to generate.

    times : array
        Array of time points (each one represents time in seconds).

    fmin : None (default) or float
        Lower cutoff frequency (in Hz). If None (default), 8 Hz are used as the
        cutoff.

    fmax : None (default) or float
        Upper cutoff frequency (in Hz). If None (default), 12 Hz are used as the
        cutoff.

    order : int, optional
        The order of the filter. By default, the order is equal to 2.

    random_state : None (default) or int
        Seed for the random number generator. If None (default), results will vary
        between function calls. Use a fixed value for reproducibility.

    Returns
    -------
    out : array, shape (n_series, n_times)
        Generated filtered white noise.
    """

    if fmin is None:
        warnings.warn("fmin was None. Setting fmin to 8 Hz", UserWarning)
        fmin = 8.0
    if fmax is None:
        warnings.warn("fmax was None. Setting fmax to 12 Hz", UserWarning)
        fmax = 12.0

    if fmin >= fmax:
        raise ValueError("fmin must be smaller than fmax.")
    if fmin <= 0 or fmax <= 0:
        raise ValueError("filter frequencies must be greater than 0")

    if not isinstance(order, int) or order <= 0:
        raise ValueError("order must be a positive integer.")

    fs = get_sfreq(times)
    rng = np.random.default_rng(seed=random_state)
    data = rng.standard_normal(size=(n_series, times.size))
    b, a = butter(N=order, Wn=np.array([fmin, fmax]) / fs * 2, btype="bandpass")
    data = filtfilt(b, a, data, axis=1)
    return normalize_variance(data)


def one_over_f_noise(n_series, times, *, slope=1, random_state=None):
    """
    Generate time series of 1/f noise with desired slope.

    Parameters
    ----------
    n_series : int
        The number of time series to generate.

    times : array
        Array of time points (each one represents time in seconds).

    slope : float, optional
        Exponent of the power-law spectrum. By default, it is equal to 1.

    random_state : None (default) or int
        Seed for the random number generator. If None (default), results will vary
        between function calls. Use a fixed value for reproducibility.

    Returns
    -------
    out : array, shape (n_series, n_times)
        Generated 1/f noise.
    """

    data = cn.powerlaw_psd_gaussian(
        slope, size=(n_series, times.size), random_state=random_state
    )
    return normalize_variance(data)


def white_noise(n_series, times, *, random_state=None):
    """
    Generate time series of white noise (e.g., to use for modeling
    measurement noise in sensor space data)

    Parameters
    ----------
    n_series : int
        Number of time series to generate.

    times : array
        Array of time points (each one represents time in seconds).

    random_state : None (default) or int
        Seed for the random number generator. If None (default), results will vary
        between function calls. Use a fixed value for reproducibility.

    Returns
    -------
    out : array, shape (n_series, n_times)
        Generated white noise.
    """

    rng = np.random.default_rng(seed=random_state)
    data = rng.standard_normal(size=(n_series, times.size))
    return normalize_variance(data)
