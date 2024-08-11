import numpy as np
import pytest

from meegsim.coupling import constant_phase_shift, ppc_von_mises
from meegsim.waveform import white_noise
from meegsim.utils import get_sfreq, theoretical_plv
from scipy.signal import hilbert
from harmoni.extratools import compute_plv


def prepare_inputs():
    n_series = 2
    fs = 1000
    times = np.arange(0, 1, 1 / fs)
    return n_series, len(times), times


def test_estimate_phase_locking_perfect_locking():
    angle1 = np.array([0, np.pi / 2, np.pi])
    angle2 = np.array([0, np.pi / 2, np.pi])
    cplv = compute_plv(angle1, angle2, m=1, n=1, plv_type='complex')

    assert np.isclose(np.abs(cplv), 1.0), f"Expected 1.0, but got {cplv}"
    assert np.isclose(np.angle(cplv), 0.0), f"Expected 0.0, but got {cplv}"


def test_estimate_phase_locking_no_locking():
    n_series, n_times, times = prepare_inputs()
    noise = white_noise(n_series, times)

    angle1 = np.angle(hilbert(noise[0]))
    angle2 = np.angle(hilbert(noise[1]))
    cplv = compute_plv(angle1, angle2, m=1, n=1, plv_type='complex')

    assert abs(cplv) <= 0.2, f"Test failed: plv between white noise is bigger than 0.2. plv = {abs(cplv)}"


@pytest.mark.parametrize("phase_lag", [np.pi / 4, np.pi / 3, np.pi / 2, np.pi, 2 * np.pi])
def test_constant_phase_shift(phase_lag):
    # Test with a simple sinusoidal waveform
    _, _, times = prepare_inputs()
    waveform = np.sin(2 * np.pi * 10 * times)

    result = constant_phase_shift(waveform, phase_lag)

    waveform = hilbert(waveform)
    result = hilbert(result)

    cplv = compute_plv(waveform, result, m=1, n=1, plv_type='complex')
    plv = np.abs(cplv)
    test_angle = np.angle(cplv)

    assert plv >= 0.99, f"Test failed: plv is smaller than 0.99. plv = {plv}"
    assert (np.abs(test_angle) - phase_lag) <= 0.01, f"Test failed: angle is different from phase_lag. difference = {np.round((np.abs(test_angle) - phase_lag),2)}"

@pytest.mark.parametrize("m, n", [
    (2, 1),
    (3, 1),
    (3/2, 1)
])
def test_different_harmonics(m, n):
    # Test with different m and n harmonics
    _, _, times = prepare_inputs()
    waveform = np.sin(2 * np.pi * 10 * times)
    phase_lag = np.pi / 3

    result = constant_phase_shift(waveform, phase_lag, m=m, n=n)

    waveform = hilbert(waveform)
    result = hilbert(result)

    cplv = compute_plv(waveform, result, m=m, n=n, plv_type='complex')
    plv = np.abs(cplv)
    test_angle = np.angle(cplv)

    assert plv >= 0.9, f"Test failed: plv is smaller than 0.9. plv = {plv}"
    assert (np.abs(test_angle) - phase_lag) <= 0.1, f"Test failed: angle is different from phase_lag. difference = {np.round((np.abs(test_angle) - phase_lag),2)}"


@pytest.mark.parametrize("kappa", [0.001, 0.1, 0.5, 1, 5, 10, 50])
def test_ppc_von_mises(kappa):
    # Test kappas that are reliable (more than 0.5)
    _, _, times = prepare_inputs()
    waveform = np.sin(2 * np.pi * 10 * times)
    phase_lag = 0

    result = ppc_von_mises(waveform, get_sfreq(times), phase_lag, kappa=kappa)

    waveform = hilbert(waveform)
    result = hilbert(result)

    cplv = compute_plv(waveform, result, m=1, n=1, plv_type='complex')
    plv = np.abs(cplv)
    plv_theoretical = theoretical_plv(kappa)

    assert plv >= plv_theoretical, f"Test failed: plv is smaller than theoretical. plv = {plv}, plv_theoretical = {plv_theoretical}"


@pytest.mark.parametrize("m, n", [
    (2, 1),
    (3, 1),
    (5/2, 1)
])
def test_ppc_von_mises_harmonics(m, n):
    # Test with different m and n harmonics
    _, _, times = prepare_inputs()
    waveform = np.sin(2 * np.pi * 10 * times)
    phase_lag = 0
    kappa = 10

    result = ppc_von_mises(waveform, get_sfreq(times), phase_lag, m=m, n=n, kappa=kappa)

    waveform = hilbert(waveform)
    result = hilbert(result)

    cplv = compute_plv(waveform, result, m=m, n=n, plv_type='complex')
    plv = np.abs(cplv)
    test_angle = np.angle(cplv)

    assert plv >= 0.8, f"Test failed: plv is smaller than 0.8. plv = {plv}"
    assert (np.abs(test_angle) - phase_lag) <= 0.1, f"Test failed: angle is different from phase_lag. difference = {np.round((np.abs(test_angle) - phase_lag),2)}"


def test_reproducibility_with_random_state():
    # Test that using a fixed random state gives the same result
    _, _, times = prepare_inputs()
    waveform = np.sin(2 * np.pi * 5 * times)
    phase_lag = np.pi / 4
    random_state = 42

    result1 = ppc_von_mises(waveform, get_sfreq(times), phase_lag, random_state=random_state)
    result2 = ppc_von_mises(waveform, get_sfreq(times), phase_lag, random_state=random_state)

    # Test that results are identical
    np.testing.assert_array_almost_equal(result1, result2)

