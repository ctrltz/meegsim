"""
Methods for setting the coupling between two signals
"""

import numpy as np

from scipy.stats import vonmises
from scipy.signal import butter, filtfilt, hilbert


def constant_phase_shift(waveform, sfreq, phase_lag, m=1, n=1, random_state=None):
    """
    Generate time series that is phase coupled to input time series. Phase coupling is deterministic.

    Parameters
    ----------
    waveform : array-like
        The input signal to be processed. It can be a real or complex time series.

    sfreq: float
        Sampling frequency of the signal, in Hz. This argument is not used in this
        function but is accepted for consistency with other coupling methods.

    phase_lag : float
        Constant phase lag to apply to the waveform in radians.

    m : int, optional
        Harmonic of interest for phase coupling, default = 1.

    n : int, optional
        Base frequency harmonic, default = 1.

    random_state: None, optional
        This parameter is accepted for consistency with other coupling functions
        but not used since no randomness is involved.

    Returns
    -------
    out : ndarray, shape (n_times,)
        The phase-coupled waveform.
    """
    if not np.iscomplexobj(waveform):
        waveform = hilbert(waveform)

    waveform_amp = np.abs(waveform)
    waveform_angle = np.angle(waveform)
    waveform_coupled = waveform_amp * np.exp(1j * m / n * waveform_angle + 1j * phase_lag)
    return np.real(waveform_coupled)


def ppc_von_mises(waveform, sfreq, phase_lag, kappa, fmin, fmax, m=1, n=1, random_state=None):
    """
    Generate time series that is phase coupled to input time series. Phase coupling is probabilistic
    and based on von Mises distribution.
    This function can be used to set up both within-frequency (1:1) and cross-frequency (n:m) coupling.

    Parameters
    ----------
    waveform : array-like
        The input signal to be processed. It can be a real or complex time series.

    sfreq : float
        Sampling frequency (in Hz).

    phase_lag : float
        Average phase lag to apply to the waveform in radians.

    kappa : float
        Concentration parameter of the von Mises distribution. With higher kappa, phase angles
        are more concentrated around the mean direction, which translates to the coupled waveform having phase shifts
        that are consistently close to the specified phase_lag. With lower kappa, phase lags along the time series
        will vary substantially. default = 1.

    fmin: float
        Lower cutoff frequency of the base frequency harmonic (in Hz). default = None.

    fmax: float
        Upper cutoff frequency of the base frequency harmonic (in Hz). default = None.

    m : int, optional
        Harmonic of interest for phase coupling, default = 1.

    n : int, optional
        Base frequency harmonic, default = 1.

    random_state : int or None, optional
        Seed for the random number generator. If None, it will be drawn
        automatically, and results will vary between function calls. Used for
        reproducibility of results when `kappa` is specified. default = None.

    Returns
    -------
    out : ndarray, shape (n_times,)
        The phase-coupled waveform.
    """

    if not np.iscomplexobj(waveform):
        waveform = hilbert(waveform)

    waveform_amp = np.abs(waveform)
    waveform_angle = np.angle(waveform)
    n_samples = len(waveform)

    ph_distr = vonmises.rvs(kappa, loc=phase_lag, size=n_samples, random_state=random_state)
    tmp_waveform = np.real(waveform_amp * np.exp(1j * m / n * waveform_angle + 1j * ph_distr))
    b, a = butter(N=2, Wn=np.array([m / n * fmin, m / n * fmax]) / sfreq * 2, btype='bandpass')
    tmp_waveform = filtfilt(b, a, tmp_waveform)
    waveform_coupled = waveform_amp * np.exp(1j * np.angle(hilbert(tmp_waveform)))

    return np.real(waveform_coupled)
