import numpy as np
import pytest

from harmoni.extratools import compute_plv
from mock import patch
from scipy.signal import hilbert

from meegsim.coupling import (
    constant_phase_shift,
    ppc_von_mises,
    ppc_shifted_copy_with_noise,
    _get_required_snr,
    _shifted_copy_with_noise,
)
from meegsim.utils import get_sfreq, theoretical_plv

from utils.prepare import prepare_sinusoid


def prepare_inputs():
    n_series = 2
    fs = 1000
    times = np.arange(0, 1, 1 / fs)
    return n_series, len(times), times


@pytest.mark.parametrize(
    "phase_lag", [np.pi / 4, np.pi / 3, np.pi / 2, np.pi, 2 * np.pi]
)
def test_constant_phase_shift(phase_lag):
    # Test with a simple sinusoidal waveform
    _, _, times = prepare_inputs()
    waveform = np.sin(2 * np.pi * 10 * times)

    result = constant_phase_shift(waveform, get_sfreq(times), phase_lag)

    waveform = hilbert(waveform)
    result = hilbert(result)

    cplv = compute_plv(waveform, result, m=1, n=1, plv_type="complex")
    plv = np.abs(cplv)
    test_angle = np.angle(cplv)

    assert plv >= 0.99, f"Test failed: plv is smaller than 0.99. plv = {plv}"
    assert (
        (np.abs(test_angle) - phase_lag) <= 0.01
    ), f"Test failed: angle is different from phase_lag. difference = {np.round((np.abs(test_angle) - phase_lag),2)}"


@pytest.mark.parametrize("m, n", [(2, 1), (3, 1), (5 / 2, 1)])
def test_constant_phase_shift_harmonics(m, n):
    # Test with different m and n harmonics
    _, _, times = prepare_inputs()
    waveform = np.sin(2 * np.pi * 10 * times)
    phase_lag = np.pi / 3

    result = constant_phase_shift(waveform, get_sfreq(times), phase_lag, m=m, n=n)

    waveform = hilbert(waveform)
    result = hilbert(result)

    cplv = compute_plv(waveform, result, m=m, n=n, plv_type="complex")
    plv = np.abs(cplv)
    test_angle = np.angle(cplv)

    assert plv >= 0.9, f"Test failed: plv is smaller than 0.9. plv = {plv}"
    assert (
        (np.abs(test_angle) - phase_lag) <= 0.1
    ), f"Test failed: angle is different from phase_lag. difference = {np.round((np.abs(test_angle) - phase_lag),2)}"


@pytest.mark.parametrize("kappa", [0.001, 0.1, 0.5, 1, 5, 10, 50])
def test_ppc_von_mises(kappa):
    # Test kappas that are reliable (more than 0.5)
    _, _, times = prepare_inputs()
    waveform = np.sin(2 * np.pi * 10 * times)
    phase_lag = 0

    result = ppc_von_mises(
        waveform, get_sfreq(times), phase_lag, kappa=kappa, fmin=8, fmax=12
    )

    waveform = hilbert(waveform)
    result = hilbert(result)

    cplv = compute_plv(waveform, result, m=1, n=1, plv_type="complex")
    plv = np.abs(cplv)
    plv_theoretical = theoretical_plv(kappa)

    assert (
        plv >= plv_theoretical
    ), f"Test failed: plv is smaller than theoretical. plv = {plv}, plv_theoretical = {plv_theoretical}"


@pytest.mark.parametrize("m, n", [(2, 1), (3, 1), (5 / 2, 1)])
def test_ppc_von_mises_harmonics(m, n):
    # Test with different m and n harmonics
    _, _, times = prepare_inputs()
    waveform = np.sin(2 * np.pi * 10 * times)
    phase_lag = 0
    kappa = 10

    result = ppc_von_mises(
        waveform, get_sfreq(times), phase_lag, m=m, n=n, kappa=kappa, fmin=8, fmax=12
    )

    waveform = hilbert(waveform)
    result = hilbert(result)

    cplv = compute_plv(waveform, result, m=m, n=n, plv_type="complex")
    plv = np.abs(cplv)
    test_angle = np.angle(cplv)

    assert plv >= 0.8, f"Test failed: plv is smaller than 0.8. plv = {plv}"
    assert (
        (np.abs(test_angle) - phase_lag) <= 0.1
    ), f"Test failed: angle is different from phase_lag. difference = {np.round((np.abs(test_angle) - phase_lag),2)}"


@pytest.mark.parametrize(
    "coupling_fun,params",
    [
        (ppc_von_mises, dict(kappa=1, phase_lag=np.pi / 4, fmin=8, fmax=12)),
        (
            ppc_shifted_copy_with_noise,
            dict(coh=0.5, phase_lag=np.pi / 4, fmin=8, fmax=12),
        ),
    ],
)
def test_reproducibility_with_random_state(coupling_fun, params):
    # Test that using a fixed random state gives the same result
    sfreq = 100
    waveform = prepare_sinusoid(f=10, sfreq=sfreq, duration=30)
    random_state = 42

    result1 = coupling_fun(
        waveform,
        sfreq,
        **params,
        random_state=random_state,
    )
    result2 = coupling_fun(
        waveform,
        sfreq,
        **params,
        random_state=random_state,
    )

    # Test that results are identical
    np.testing.assert_array_almost_equal(result1, result2)


@pytest.mark.parametrize(
    "coh,expected_snr",
    [
        (1.0, np.inf),
        (1.0 / np.sqrt(2), 1.0),
        (0.0, 0.0),
    ],
)
def test_get_required_snr(coh, expected_snr):
    assert np.isclose(_get_required_snr(coh), expected_snr)


@pytest.mark.parametrize(
    "target_coh,tol",
    [
        # NOTE: we increase tolerance for low values of coherence since
        # the variance might increase
        (0.9, 0.1),
        (0.7, 0.15),
        (0.5, 0.2),
        (0.3, 0.25),
    ],
)
def test_ppc_shifted_copy_with_noise(target_coh, tol):
    sfreq = 100
    waveform = prepare_sinusoid(f=10, sfreq=sfreq, duration=120)
    seed = 1234

    phase_lag = 0
    coupled = ppc_shifted_copy_with_noise(
        waveform=waveform,
        sfreq=sfreq,
        phase_lag=phase_lag,
        coh=target_coh,
        fmin=8,
        fmax=12,
        random_state=seed,
    )
    actual_coh = compute_plv(waveform, coupled, m=1, n=1, coh=True)
    assert np.abs(actual_coh - target_coh) < tol

    # Coupling should work regardless of the phase lag
    phase_lag = np.pi / 2
    coupled = ppc_shifted_copy_with_noise(
        waveform=waveform,
        sfreq=sfreq,
        phase_lag=phase_lag,
        coh=target_coh,
        fmin=8,
        fmax=12,
        random_state=seed,
    )
    actual_coh = compute_plv(waveform, coupled, m=1, n=1, coh=True)
    assert np.abs(actual_coh - target_coh) < tol


def test_shifted_copy_with_noise_infinite_snr():
    sfreq = 100
    # Multiply with the square root of 2 to normalize the variance
    waveform = np.sqrt(2) * prepare_sinusoid(f=10, sfreq=sfreq, duration=30)

    # Infinite SNR with no phase lag should return the input
    coupled = _shifted_copy_with_noise(waveform, sfreq, 0, np.inf, 8, 12, None)
    assert np.allclose(coupled, waveform)


@patch("meegsim.coupling.narrowband_oscillation", return_value=np.ones((100,)))
def test_shifted_copy_with_noise_zero_snr(osc_mock):
    sfreq = 100
    waveform = np.sqrt(2) * prepare_sinusoid(f=10, sfreq=sfreq, duration=30)

    # Zero SNR should return noise waveform (mock in our case)
    coupled = _shifted_copy_with_noise(waveform, sfreq, 0, 0, 8, 12, None)
    assert np.allclose(coupled, 1.0)
    osc_mock.assert_called_once()
