import numpy as np
import mne
import warnings

from scipy.signal import butter, filtfilt

from meegsim.sources import _combine_sources_into_stc


def get_sensor_space_variance(stc, fwd, fmin=None, fmax=None, filter=False):
    """
    Estimate the sensor space variance of the provided stc

    Parameters
    ----------
    stc: mne.SourceEstimate
        Source estimate containing signal or noise (vertices x times).

    fwd: mne.Forward
        Forward model.

    fmin: float, optional
        Lower cutoff frequency (in Hz). default = None.

    fmax: float, optional
        Upper cutoff frequency (in Hz). default = None.

    filter: bool, optional
        Indicate if filtering in the band of oscillations is required. default = False.

    Returns
    -------
    stc_var: float
        Sensor space variance in the frequency band, if specified.
    """

    try:
        raw = mne.apply_forward_raw(fwd, stc, fwd["info"], on_missing="raise")
    except ValueError:
        raise ValueError(
            "The provided forward model does not contain some of the "
            "simulated sources, so the SNR cannot be adjusted."
        )

    raw_data = raw.get_data()
    if filter:
        if fmin is None or fmax is None:
            raise ValueError(
                "Frequency band limits are required for the adjustment of SNR."
            )

        b, a = butter(2, np.array([fmin, fmax]) / stc.sfreq * 2, btype="bandpass")
        raw_data = filtfilt(b, a, raw_data, axis=1)

    n_samples = raw_data.shape[1]
    sensor_cov = (raw_data @ raw_data.T) / (n_samples - 1)
    sensor_var = np.mean(np.diag(sensor_cov))

    return sensor_var


def amplitude_adjustment_factor(signal_var, noise_var, target_snr):
    """
    Derive the adjustment factor for signal amplitude that allows obtaining the target SNR

    Parameters
    ----------
    signal_var: float
        Variance of the simulated signal with respect to leadfield. Can be obtained with
        a function snr.get_sensor_space_variance.

    noise_var: float
        Variance of the simulated noise with respect to leadfield. Can be obtained with
        a function snr.get_sensor_space_variance.

    target_snr: float
        Value of a desired SNR for the signal.

    Returns
    -------
    factor: float
        The original signal should be multiplied by this value to obtain the desired SNR.
    """

    snr_current = np.divide(signal_var, noise_var)

    if np.isinf(snr_current):
        raise ValueError(
            "The noise variance appears to be zero, so the initial SNR "
            "cannot be calculated. Please check the created noise."
        )

    factor = np.sqrt(target_snr / snr_current)

    if np.isinf(factor):
        raise ValueError(
            "The signal variance and thus the initial SNR appear to be "
            "zero, so SNR cannot be adjusted. Please check the created "
            "signals."
        )

    return factor


def _adjust_snr_local(src, fwd, tstep, sources, source_groups, noise_sources):
    """
    Perform the adjustment of local SNR: the power of each source is adjusted
    relative to the power of all noise sources combined. The waveforms are
    adjusted in-place.
    """
    # Get the stc and leadfield of all noise sources
    if not noise_sources:
        raise ValueError(
            "No noise sources were added to the simulation, so the SNR "
            "cannot be adjusted."
        )
    stc_noise = _combine_sources_into_stc(noise_sources.values(), src, tstep)

    # Adjust the SNR of sources in each source group
    for sg in source_groups:
        if sg.snr is None:
            continue

        # Estimate the noise variance in the specified frequency band
        fmin, fmax = sg.snr_params["fmin"], sg.snr_params["fmax"]
        noise_var = get_sensor_space_variance(
            stc_noise, fwd, fmin=fmin, fmax=fmax, filter=True
        )

        # Adjust the amplitude of each source in the group to match the target SNR
        for name, target_snr in zip(sg.names, sg.snr):
            s = sources[name]

            # NOTE: taking a safer approach for now and filtering
            # even if the signal is already a narrowband oscillation
            signal_var = get_sensor_space_variance(
                s.to_stc(src, tstep), fwd, fmin=fmin, fmax=fmax, filter=True
            )

            # NOTE: patch sources might require more complex calculations
            # if the within-patch correlation is not equal to 1
            factor = amplitude_adjustment_factor(signal_var, noise_var, target_snr)
            s.waveform *= factor


def _adjust_snr_global(src, fwd, snr_global, snr_params, tstep, sources, noise_sources):
    """
    Perform the adjustment of global SNR: the power of all sources combined
    is adjusted relative to the power of all noise sources combined. The
    waveforms are adjusted in-place.
    """
    # Combine signal/noise sources
    if not sources:
        warnings.warn(
            "No point/patch sources were added to the simulation, "
            "skipping the requested adjustment of global SNR."
        )
        return

    stc_signal = _combine_sources_into_stc(sources.values(), src, tstep)

    if not noise_sources:
        raise ValueError(
            "No noise sources were added to the simulation, so the global SNR "
            "cannot be adjusted."
        )
    stc_noise = _combine_sources_into_stc(noise_sources.values(), src, tstep)

    # Get sensor-space variance of signal and noise
    fmin, fmax = snr_params["fmin"], snr_params["fmax"]
    noise_var = get_sensor_space_variance(
        stc_noise, fwd, fmin=fmin, fmax=fmax, filter=True
    )
    signal_var = get_sensor_space_variance(
        stc_signal, fwd, fmin=fmin, fmax=fmax, filter=True
    )

    # Adjust the amplitudes of all sources by the same (!) factor
    factor = amplitude_adjustment_factor(signal_var, noise_var, snr_global)
    for s in sources.values():
        s.waveform *= factor
