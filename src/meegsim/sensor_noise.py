import numpy as np

from .snr import amplitude_adjustment_factor
from .waveform import white_noise


def _prepare_sensor_noise(raw, times, random_state):
    n_chans = len(raw.ch_names)
    noise = white_noise(n_chans, times,
                        random_state=random_state)
    
    # Scale the noise to equalize the mean sensor-space variance of 
    # brain activity and sensor noise
    signal = raw.get_data()
    signal_var = np.trace(signal @ signal.T)
    noise_var = np.trace(noise @ noise.T)
    factor = amplitude_adjustment_factor(signal_var, noise_var, target_snr=1)

    # Dividing here since we are adjusting noise, not the signal
    noise /= factor

    return noise


def _adjustment_factors(noise_level):
    return np.sqrt(1 - noise_level), np.sqrt(noise_level)


def _adjust_sensor_noise(raw, noise, sensor_noise_level):
    """
    Mix brain activity and sensor noise together
    sensor_noise_level = power_sensor_noise / power_total
    """

    factor_signal, factor_noise = _adjustment_factors(sensor_noise_level)
    
    raw_mixed = raw.copy()
    raw_mixed._data = factor_signal * raw._data + factor_noise * noise

    return raw_mixed
