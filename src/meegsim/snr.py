import numpy as np
import warnings

def get_sensor_space_variance(stc, fwd, *, fmin=None, fmax=None, filter=False):
    """
    Estimate the sensor space variance of the provided stc

    Parameters
    ----------
    stc: mne.SourceEstimate
        Source estimate containing with signal or noise (vertices x times).

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
        Variance with respect to leadfield.
    """
    from scipy.signal import butter, filtfilt
    leadfield = fwd['sol']['data']
    if filter:
        if fmin is None:
            warnings.warn("fmin was None. Setting fmin to 8 Hz", UserWarning)
            fmin = 8.
        if fmax is None:
            warnings.warn("fmax was None. Setting fmax to 12 Hz", UserWarning)
            fmax = 12.
        b, a = butter(2, np.array([fmin, fmax]) / stc.sfreq * 2, btype='bandpass')
        stc_data = filtfilt(b, a, stc._data, axis=1)
    else:
        stc_data = stc._data

    nonzero_idx = np.mean(stc_data, axis=1) > 0
    stc_var = np.mean(stc_data[nonzero_idx, :] ** 2) * np.mean(leadfield[:, nonzero_idx] ** 2)
    return stc_var


def adjust_snr(signal_var, noise_var, *, target_snr=1):
    """
    Derive the signal amplitude that allows obtaining target SNR

    Parameters
    ----------
    signal_var: float
        Variance of the simulated signal with respect to leadfield. Can be obtained with
        a function snr.get_sensor_space_variance.

    noise_var: float
        Variance of the simulated noise with respect to leadfield. Can be obtained with
        a function snr.get_sensor_space_variance.

    target_snr: float, optional
        Value of a desired SNR for the signal. default = 1.

    Returns
    -------
    out: float
        The value that original signal should be scaled (divided) to in order to obtain desired SNR.
    """
    snr_current = signal_var / noise_var
    return np.sqrt(snr_current / target_snr)
