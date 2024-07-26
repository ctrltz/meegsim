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

import numpy as np

from .utils import normalize_power


def narrowband_oscillation(n_series, times, *, fmin=8, fmax=12, order=2, random_state=None):
    # TODO: 
    #  - move filtered_randn here (without hilbert)
    #  - NB: divide by the norm to make sure that signal power is always the same
    #  - probably we should check the provided values for fmin/fmax/order
    #
    # Ideas for tests
    #
    # Test 1 (fmin/fmax)
    #  - generate the time series and calculate their spectra
    #  - sort frequencies according to their power descending
    #  - frequencies within the specified band should be among the first ones
    # 
    # Test 2 (order)
    #  - we could check that we pass correct value to filtfilt (requires mocking)
    #
    # Test 3 (random_state)
    #  - fix random_state, call the function twice and compare (see np.allclose)
    #  - check that results are different when random_state is None
    #
    # In all tests: check the shape of the resulting array
    pass
    
    
def one_over_f_noise(n_series, times, *, slope=1, random_state=None):
    # TODO: divide by the norm to make sure the all amplitudes are equal 
    pass
    

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