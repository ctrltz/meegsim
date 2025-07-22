"""
Methods for setting the coupling between two signals
"""

import numpy as np

from scipy.stats import vonmises
from scipy.signal import butter, filtfilt, hilbert

from meegsim._check import check_numeric, check_option
from meegsim.snr import get_variance, amplitude_adjustment_factor
from meegsim.utils import normalize_variance
from meegsim.waveform import narrowband_oscillation, white_noise


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
        generated (default) or to be the same as the envelope of the input waveform.

    fmin : float, optional
        Lower cutoff frequency for the oscillation that gives rise to the random
        amplitude envelope (only if the ``envelope`` is set to ``"random"``).

    fmax : float, optional
        Upper cutoff frequency for the oscillation that gives rise to the random
        amplitude envelope (only if the ``envelope`` is set to ``"random"``).

    m : float, optional
        Multiplier for the base frequency of the output oscillation, default is 1.

    n : float, optional
        Multiplier for the base frequency of the input oscillation, default is 1.

    random_state : None, optional
        Random state can be fixed to provide reproducible results if the envelope
        is generated randomly. Otherwise, the results may differ between function calls.

    Returns
    -------
    out : array, shape (n_times,)
        The phase-coupled waveform with the same amplitude envelope as the input one.
    """
    if not np.iscomplexobj(waveform):
        waveform = hilbert(waveform)

    waveform_amp = _get_envelope(waveform, envelope, sfreq, fmin, fmax, random_state)
    waveform_angle = np.angle(waveform)
    waveform_coupled = np.real(
        waveform_amp * np.exp(1j * m / n * waveform_angle + 1j * phase_lag)
    )
    if envelope == "same":
        return normalize_variance(waveform_coupled)

    # NOTE: if the envelope was modified, we filter the result again in the target
    # frequency range to suppress possible distortions due to merging amplitude
    # envelope and phase from different time series
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
        generated (default) or to be the same as the envelope of the input waveform.

    m : float, optional
        Multiplier for the base frequency of the output oscillation, default is 1.

    n : float, optional
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

    # NOTE: we filter the result again in the target frequency range to suppress
    # possible distortions due to separate adjustment of the phase and amplitude
    # of the coupled time series
    b, a = butter(
        N=2, Wn=np.array([m / n * fmin, m / n * fmax]) / sfreq * 2, btype="bandpass"
    )
    waveform_coupled = filtfilt(b, a, waveform_coupled)

    return normalize_variance(waveform_coupled)


def _shifted_copy_with_noise(
    waveform, sfreq, phase_lag, snr, fmin, fmax, band_limited, random_state
):
    """
    Generate a coupled time series by (1) applying a constant phase shift to the input
    waveform and (2) mixing it with noise to achieve a desired level of signal-to-noise
    ratio, which determines the resulting phase-phase and amplitude-amplitude coupling.
    """
    shifted_waveform = ppc_constant_phase_shift(
        waveform, sfreq, phase_lag, envelope="same"
    )
    signal_var = get_variance(shifted_waveform, sfreq, fmin, fmax, filter=True)

    # NOTE: to make coupling band-limited (substantial only in the band of interest),
    # we need to corrupt the rest of the coherence spectra with white noise,
    # affecting other parts of the signal apart from the frequency band of interest.
    #
    # For oscillations as our main case this is not a big deal but might be important
    # for other signals. If we filter the added noise in the frequency band of
    # interest, it leads to flat connectivity spectra but only affects target frequencies
    times = np.arange(waveform.size) / sfreq
    if band_limited:
        noise_waveform = white_noise(n_series=1, times=times, random_state=random_state)
    else:
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


def _get_required_snr(coh, band_limited):
    """
    Calculate the value of SNR that is required to obtain desired coherence
    between a waveform and its copy mixed with noise.
    """
    # NOTE: prevent infinite SNR to always mix some noise in case we need to make
    # the coupling band-limited
    if band_limited and np.isclose(coh, 1, atol=1e-3):
        coh = 0.999

    return np.divide(coh**2, 1 - coh**2)


def ppc_shifted_copy_with_noise(
    waveform, sfreq, phase_lag, coh, fmin, fmax, band_limited=True, random_state=None
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

    band_limited : bool
        Whether to limit coupling only to the frequency band of interest (True by
        default). If set to False, coupling will be the same for all frequencies,
        resulting in a flat connectivity spectra. However, the signal outside of the
        frequency band of interest will be modified negligibly.

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
    snr = _get_required_snr(coh, band_limited)
    return _shifted_copy_with_noise(
        waveform=waveform,
        sfreq=sfreq,
        phase_lag=phase_lag,
        snr=snr,
        fmin=fmin,
        fmax=fmax,
        band_limited=band_limited,
        random_state=random_state,
    )
