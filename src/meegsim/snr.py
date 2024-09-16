import numpy as np
import warnings
import mne

from scipy.signal import butter, filtfilt


def get_sensor_space_variance(stc, fwd, *, fmin=None, fmax=None, filter=False):
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
        Variance with respect to leadfield.
    """

    stc_data = stc.data
    if filter:
        if fmin is None or fmax is None:
            raise ValueError(
                'Frequency band limits are required for the adjustment of SNR.'
            )

        b, a = butter(2, np.array([fmin, fmax]) / stc.sfreq * 2, btype='bandpass')
        stc_data = filtfilt(b, a, stc_data, axis=1)        

    fwd_restrict = mne.forward.restrict_forward_to_stc(fwd, stc, 
                                                       on_missing='raise')
    leadfield_restict = fwd_restrict['sol']['data']

    n_samples = stc_data.shape[1]
    n_sensors = leadfield_restict.shape[0]
    source_cov = (stc_data @ stc_data.T) / n_samples
    sensor_cov = leadfield_restict @ source_cov @ leadfield_restict.T
    sensor_var = np.trace(sensor_cov) / n_sensors

    return sensor_var


def adjust_snr(signal_var, noise_var, target_snr):
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

    target_snr: float
        Value of a desired SNR for the signal.

    Returns
    -------
    out: float
        The value that original signal should be scaled (multiplied) to in order to obtain desired SNR.
    """

    snr_current = np.divide(signal_var, noise_var)

    if np.isinf(snr_current):
        raise ValueError("The noise variance appears to be zero, so the initial SNR "
                         "cannot be calculated. Please check the created noise.")

    factor = np.sqrt(target_snr / snr_current)

    if np.isinf(factor):
        raise ValueError("The signal variance and thus the initial SNR appear to be "
                         "zero, so SNR cannot be adjusted. Please check the created "
                         "signals.")

    return factor
