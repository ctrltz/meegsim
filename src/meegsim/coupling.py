"""
Methods for setting the coupling between two signals
"""

import numpy as np

from scipy.stats import vonmises
from scipy.signal import butter, filtfilt, hilbert

from meegsim._check import check_numeric, check_option
from meegsim.snr import get_variance, amplitude_adjustment_factor
from meegsim.utils import normalize_variance
from meegsim.waveform import narrowband_oscillation


def _get_envelope(waveform, envelope, sfreq, fmin=None, fmax=None, random_state=None):
    check_option(
        "the amplitude envelope of the coupled waveform", envelope, ["same", "random"]
    )
    if not np.iscomplexobj(waveform):
        waveform = hilbert(waveform)

    if envelope == "same":
        return np.abs(waveform)

    if fmin is None or fmax is None:
        raise ValueError(
            "Frequency limits are required for generating the envelope of the coupled waveform"
        )
    times = np.arange(waveform.size) / sfreq
    random_waveform = narrowband_oscillation(
        1, times, fmin=fmin, fmax=fmax, random_state=random_state
    )
    random_waveform = hilbert(random_waveform)

    # TODO: here we could also mix original and random envelope with different
    # values of SNR to achieve smooth control over the resulting envelope correlation
    return np.abs(random_waveform)


def ppc_constant_phase_shift(
    waveform,
    sfreq,
    phase_lag,
    fmin=None,
    fmax=None,
    envelope="random",
    m=1,
    n=1,
    random_state=None,
):
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
        Sampling frequency of the signal, in Hz.

    phase_lag : float
        Constant phase lag to apply to the waveform in radians.

    envelope : str, {"same", "random"}
        Controls the amplitude envelope of the coupled waveform to be either randomly
        generated or to be the same as the envelope of the input waveform.

    fmin : float, optional
        Lower cutoff frequency for the oscillation.

    fmax : float, optional
        Upper cutoff frequency for the oscillation.

    m : float, optional
        Multiplier for the base frequency of the output oscillation, default is 1.

    n : float, optional
        Multiplier for the base frequency of the input oscillation, default is 1.

    random_state : None, optional
        Random state can be fixed to provide reproducible results. Otherwise, the
        results may differ between function calls.

    Returns
    -------
    out : array, shape (n_times,)
        The phase-coupled waveform with the same amplitude envelope as the input one.
    """
    if not np.iscomplexobj(waveform):
        waveform = hilbert(waveform)
    waveform_angle = np.angle(waveform)

    waveform_amp = _get_envelope(waveform, envelope, sfreq, fmin, fmax, random_state)
    waveform_coupled = np.real(
        waveform_amp * np.exp(1j * m / n * waveform_angle + 1j * phase_lag)
    )
    if envelope == "same":
        return normalize_variance(waveform_coupled)

    # NOTE: if the envelope was modified, we filter the result again in the target
    # frequency range to suppress distortions due to merging amplitude envelope and
    # phase from different time series
    b, a = butter(
        N=2, Wn=np.array([m / n * fmin, m / n * fmax]) / sfreq * 2, btype="bandpass"
    )
    waveform_coupled = filtfilt(b, a, waveform_coupled)

    return normalize_variance(waveform_coupled)


def ppc_von_mises(
    waveform,
    sfreq,
    phase_lag,
    kappa,
    fmin,
    fmax,
    envelope="random",
    m=1,
    n=1,
    random_state=None,
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

    envelope : str, {"same", "random"}
        Controls the amplitude envelope of the coupled waveform to be either randomly
        generated or to be the same as the envelope of the input waveform.

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

    waveform_amp = _get_envelope(waveform, envelope, sfreq, fmin, fmax, random_state)
    waveform_angle = np.angle(waveform)
    n_samples = waveform.size

    ph_distr = vonmises.rvs(
        kappa, loc=phase_lag, size=n_samples, random_state=random_state
    )
    waveform_coupled = np.real(
        waveform_amp * np.exp(1j * m / n * waveform_angle + 1j * ph_distr)
    )
    b, a = butter(
        N=2, Wn=np.array([m / n * fmin, m / n * fmax]) / sfreq * 2, btype="bandpass"
    )
    waveform_coupled = filtfilt(b, a, waveform_coupled)

    return normalize_variance(waveform_coupled)


def _shifted_copy_with_noise(waveform, sfreq, phase_lag, snr, fmin, fmax, random_state):
    """
    Generate a coupled time series by (1) applying a constant phase shift to the input
    waveform and (2) mixing it with noise to achieve a desired level of signal-to-noise
    ratio, which determines the resulting phase-phase and amplitude-amplitude coupling.
    """
    shifted_waveform = ppc_constant_phase_shift(
        waveform, sfreq, phase_lag, envelope="same"
    )
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
    # SNR = inf <-> coherence = 1 -> return copy of the input
    # SNR = 0 <-> coherence = 0 -> return the generated noise
    if np.isinf(snr):
        return shifted_waveform
    if np.isclose(snr, 0):
        return noise_waveform

    factor = amplitude_adjustment_factor(signal_var, noise_var, snr)
    coupled_waveform = factor * shifted_waveform + noise_waveform

    return normalize_variance(coupled_waveform)


def _get_required_snr(coh):
    """
    Calculate the value of SNR that is required to obtain desired coherence
    between a waveform and its copy mixed with noise.
    """
    return np.divide(coh**2, 1 - coh**2)


def ppc_shifted_copy_with_noise(
    waveform, sfreq, phase_lag, coh, fmin, fmax, random_state=None
):
    """
    Generate a time series with desired level of coherence with the provided waveform
    in a frequency band of interest.

    The time series are generated by (1) applying a constant phase shift to the input
    waveform and (2) mixing it with a specific amount of narrowband noise. This
    function only supports within-frequency coupling.

    Parameters
    ----------
    waveform : array
        The input signal to be processed.

    sfreq : float
        Sampling frequency (in Hz).

    phase_lag : float
        Average phase lag to apply to the waveform in radians.

    coh : float
        The desired level of coherence between input and output time series.

    fmin : float
        Lower cutoff frequency of the frequency band of interest (in Hz).

    fmax : float
        Upper cutoff frequency of the frequency band of interest (in Hz).

    random_state : None (default) or int
        Seed for the random number generator. If None (default), results will vary
        between function calls. Use a fixed value for reproducibility.

    Returns
    -------
    out : array, shape (n_times,)
        The phase-coupled waveform.

    Notes
    -----
    The desired value of coherence and phase lags are obtained only on average across
    multiple simulations. For every individual output, coherence and phase lag might
    deviate from the desired values: The lower the requested coherence is, the higher
    is the variance of the output. For more information, see :doc:`this example
    </auto_examples/plot_coupling>`.
    """
    check_numeric("coherence", coh, bounds=(0, 1), allow_none=False)
    snr = _get_required_snr(coh)
    return _shifted_copy_with_noise(
        waveform=waveform,
        sfreq=sfreq,
        phase_lag=phase_lag,
        snr=snr,
        fmin=fmin,
        fmax=fmax,
        random_state=random_state,
    )
