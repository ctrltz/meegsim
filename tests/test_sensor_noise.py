import numpy as np
import pytest

from meegsim.sensor_noise import _prepare_sensor_noise, _adjustment_factors
from meegsim.simulate import SourceSimulator
from meegsim.waveform import narrowband_oscillation

from utils.prepare import prepare_forward, prepare_source_space


def test_prepare_sensor_noise():
    src = prepare_source_space(types=["surf", "surf"], vertices=[[0, 1], [0, 1]])
    fwd = prepare_forward(5, 4)

    sim = SourceSimulator(src)
    sim.add_noise_sources(location=[(0, 0), (1, 0)])
    sim.add_point_sources(
        location=[(0, 1), (1, 1)],
        waveform=narrowband_oscillation,
        waveform_params=dict(fmin=8, fmax=12),
    )

    sc = sim.simulate(sfreq=250, duration=30, random_state=123)
    raw = sc.to_raw(fwd, fwd["info"])
    noise = _prepare_sensor_noise(raw, sc.times, sc.random_state)

    # Check that the mean variances of brain activity and noise are equal
    signal = raw.get_data()
    signal_var = np.trace(signal @ signal.T)
    noise_var = np.trace(noise @ noise.T)
    assert np.allclose(signal_var, noise_var)


@pytest.mark.parametrize(
    "noise_level,expected_f_signal,expected_f_noise",
    [(0, 1, 0), (0.36, 0.8, 0.6), (1, 0, 1)],
)
def test_adjustment_factors(noise_level, expected_f_signal, expected_f_noise):
    f_signal, f_noise = _adjustment_factors(noise_level)
    assert np.allclose(f_signal, expected_f_signal)
    assert np.allclose(f_noise, expected_f_noise)
