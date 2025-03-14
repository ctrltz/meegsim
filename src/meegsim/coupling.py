"""
Methods for setting the coupling between two signals
"""

import numpy as np

from scipy.stats import vonmises
from scipy.signal import butter, filtfilt, hilbert

from meegsim.snr import get_variance, amplitude_adjustment_factor
from meegsim.utils import normalize_variance
from meegsim.waveform import narrowband_oscillation


def constant_phase_shift(waveform, sfreq, phase_lag, m=1, n=1, random_state=None):
    """
    Generate a time series that is phase coupled to the input time series with
    a constant phase lag.

    This function can be used to set up both within-frequency (1:1, default) and
    cross-frequency (n:m) coupling.

    .. note::
        This function is using Hilbert transform for manipulating the phase of
        the time series, so the result might not be meaningful if applied to
        broadband data.

    Parameters
    ----------
    waveform : array
        The input signal to be processed. It can be a real or complex time series.

    sfreq : float
        Sampling frequency of the signal, in Hz. This argument is not used in this
        function but is accepted for consistency with other coupling methods.

    phase_lag : float
        Constant phase lag to apply to the waveform in radians.

    m : int, optional
        Multiplier for the base frequency of the output oscillation, default is 1.

    n : int, optional
        Multiplier for the base frequency of the input oscillation, default is 1.

    random_state : None, optional
        This parameter is accepted for consistency with other coupling functions
        but not used since no randomness is involved.

    Returns
    -------
    out : array, shape (n_times,)
        The phase-coupled waveform with the same amplitude envelope as the input one.
    """
    if not np.iscomplexobj(waveform):
        waveform = hilbert(waveform)

    waveform_amp = np.abs(waveform)
    waveform_angle = np.angle(waveform)
    waveform_coupled = waveform_amp * np.exp(
        1j * m / n * waveform_angle + 1j * phase_lag
    )
    return normalize_variance(np.real(waveform_coupled))


def ppc_von_mises(
    waveform, sfreq, phase_lag, kappa, fmin, fmax, m=1, n=1, random_state=None
):
    """
    Generate a time series that is phase coupled to the input time series with
    a probabilistic phase lag based on the von Mises distribution.

    This function can be used to set up both within-frequency (1:1, default) and
    cross-frequency (n:m) coupling.

    .. note::
        This function is using Hilbert transform for manipulating the phase of
        the time series, so the result might not be meaningful if applied to
        broadband data.

    Parameters
    ----------
    waveform : array
        The input signal to be processed. It can be a real or complex time series.

    sfreq : float
        Sampling frequency (in Hz).

    phase_lag : float
        Average phase lag to apply to the waveform in radians.

    kappa : float
        Concentration parameter of the von Mises distribution. With higher kappa,
        phase shifts between input and output waveforms are more concentrated
        around the mean value provided in ``phase_lag``. With lower kappa, phase
        shifts will vary substantially for different time points.

    fmin: float
        Lower cutoff frequency of the base frequency harmonic (in Hz).

    fmax: float
        Upper cutoff frequency of the base frequency harmonic (in Hz).

    m : int, optional
        Multiplier for the base frequency of the output oscillation, default is 1.

    n : int, optional
        Multiplier for the base frequency of the input oscillation, default is 1.

    random_state : None (default) or int
        Seed for the random number generator. If None (default), results will vary
        between function calls. Use a fixed value for reproducibility.

    Returns
    -------
    out : array, shape (n_times,)
        The phase-coupled waveform with the same amplitude envelope as the input one.
    """

    if not np.iscomplexobj(waveform):
        waveform = hilbert(waveform)

    waveform_amp = np.abs(waveform)
    waveform_angle = np.angle(waveform)
    n_samples = len(waveform)

    ph_distr = vonmises.rvs(
        kappa, loc=phase_lag, size=n_samples, random_state=random_state
    )
    tmp_waveform = np.real(
        waveform_amp * np.exp(1j * m / n * waveform_angle + 1j * ph_distr)
    )
    b, a = butter(
        N=2, Wn=np.array([m / n * fmin, m / n * fmax]) / sfreq * 2, btype="bandpass"
    )
    tmp_waveform = filtfilt(b, a, tmp_waveform)
    waveform_coupled = waveform_amp * np.exp(1j * np.angle(hilbert(tmp_waveform)))

    return normalize_variance(np.real(waveform_coupled))


def _shifted_copy_with_noise(waveform, sfreq, snr, phase_lag, fmin, fmax, random_state):
    """
    Generate a coupled time series by (1) applying a constant phase shift to the input
    waveform and (2) mixing it with noise to achieve a desired level of signal-to-noise
    ratio, which determines the resulting phase-phase and amplitude-amplitude coupling.
    """
    shifted_waveform = constant_phase_shift(waveform, sfreq, phase_lag)
    signal_var = get_variance(shifted_waveform, sfreq, fmin, fmax, filter=True)

    # NOTE: we use another randomly generated narrowband oscillation as noise here
    # so that only the frequency band of interest is affected. If a broadband signal
    # is provided as input, this behavior might not be desirable (?). In this case,
    # it can be addressed with an additional parameter
    times = np.arange(waveform.size) / sfreq
    noise_waveform = narrowband_oscillation(
        n_series=1, times=times, fmin=fmin, fmax=fmax, random_state=random_state
    )
    noise_var = get_variance(noise_waveform, sfreq, fmin, fmax, filter=True)

    # Process the corner cases
    if np.isinf(snr):
        return shifted_waveform
    if np.isclose(snr, 0):
        return noise_waveform

    factor = amplitude_adjustment_factor(signal_var, noise_var, snr)
    coupled_waveform = factor * shifted_waveform + noise_waveform

    return normalize_variance(coupled_waveform)


def _get_required_snr(coh):
    return np.divide(coh**2, 1 - coh**2)


def ppc_shifted_copy_with_noise(
    waveform, sfreq, coh, phase_lag, fmin, fmax, random_state=None
):
    """
    Generate a time series with desired level of coherence with the provided waveform.
    """
    snr = _get_required_snr(coh)
    return _shifted_copy_with_noise(
        waveform=waveform,
        sfreq=sfreq,
        snr=snr,
        phase_lag=phase_lag,
        fmin=fmin,
        fmax=fmax,
        random_state=random_state,
    )
