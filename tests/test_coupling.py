import numpy as np
import pytest

from harmoni.extratools import compute_plv
from mock import patch
from scipy.signal import hilbert

from meegsim.coupling import (
    ppc_constant_phase_shift,
    ppc_von_mises,
    ppc_shifted_copy_with_noise,
    _get_envelope,
    _get_required_snr,
    _shifted_copy_with_noise,
)
from meegsim.utils import get_sfreq

from utils.prepare import prepare_sinusoid


def prepare_inputs(sfreq=250, duration=60):
    n_series = 2
    times = np.arange(0, duration, 1 / sfreq)
    return n_series, len(times), times


@pytest.mark.parametrize(
    "phase_lag", [np.pi / 4, np.pi / 3, np.pi / 2, np.pi, 2 * np.pi]
)
def test_ppc_constant_phase_shift_same_envelope(phase_lag):
    # Test with a simple sinusoidal waveform
    _, _, times = prepare_inputs()
    waveform = np.sin(2 * np.pi * 10 * times)

    result = ppc_constant_phase_shift(
        waveform, get_sfreq(times), phase_lag, envelope="same", random_state=1234
    )

    waveform = hilbert(waveform)
    result = hilbert(result)

    cplv = compute_plv(waveform, result, m=1, n=1, plv_type="complex")
    plv = np.abs(cplv)
    test_angle = np.angle(cplv)

    assert plv >= 0.9, f"Expected PLV to be at least 0.9, got {plv}"
    assert (
        (np.abs(test_angle) - phase_lag) <= 0.01
    ), f"Test failed: angle is different from phase_lag. difference = {np.round((np.abs(test_angle) - phase_lag),2)}"


@pytest.mark.parametrize(
    "phase_lag", [np.pi / 4, np.pi / 3, np.pi / 2, np.pi, 2 * np.pi]
)
def test_ppc_constant_phase_shift_random_envelope(phase_lag):
    # Test with a simple sinusoidal waveform
    _, _, times = prepare_inputs()
    waveform = np.sin(2 * np.pi * 10 * times)

    result = ppc_constant_phase_shift(
        waveform,
        get_sfreq(times),
        phase_lag,
        envelope="random",
        fmin=9.5,
        fmax=10.5,
        random_state=1234,
    )

    waveform = hilbert(waveform)
    result = hilbert(result)

    cplv = compute_plv(waveform, result, m=1, n=1, plv_type="complex")
    plv = np.abs(cplv)
    test_angle = np.angle(cplv)

    assert plv >= 0.9, f"Expected PLV to be at least 0.9, got {plv}"
    assert (
        (np.abs(test_angle) - phase_lag) <= 0.01
    ), f"Test failed: angle is different from phase_lag. difference = {np.round((np.abs(test_angle) - phase_lag),2)}"


@pytest.mark.parametrize("m, n", [(2, 1), (3, 1), (5 / 2, 1)])
def test_ppc_constant_phase_shift_harmonics_same_envelope(m, n):
    # Test with different m and n harmonics
    _, _, times = prepare_inputs()
    waveform = np.sin(2 * np.pi * 10 * times)
    phase_lag = np.pi / 3

    result = ppc_constant_phase_shift(
        waveform,
        get_sfreq(times),
        phase_lag,
        envelope="same",
        m=m,
        n=n,
        random_state=1234,
    )

    waveform = hilbert(waveform)
    result = hilbert(result)

    cplv = compute_plv(waveform, result, m=m, n=n, plv_type="complex")
    plv = np.abs(cplv)
    test_angle = np.angle(cplv)

    assert plv >= 0.9, f"Expected PLV to be at least 0.9, got {plv}"
    assert (
        (np.abs(test_angle) - phase_lag) <= 0.1
    ), f"Test failed: angle is different from phase_lag. difference = {np.round((np.abs(test_angle) - phase_lag),2)}"


@pytest.mark.parametrize(
    "kappa,lo,hi",
    [
        (0.001, 0, 0.2),
        (0.01, 0, 0.2),
        (0.1, 0.1, 0.4),
        (0.3, 0.3, 0.7),
        (0.5, 0.5, 0.9),
        (1, 0.7, 1),
        (10, 0.9, 1),
    ],
)
def test_ppc_von_mises_same_envelope_kappa(kappa, lo, hi):
    _, _, times = prepare_inputs(duration=120)
    waveform = np.sin(2 * np.pi * 10 * times)
    phase_lag = np.pi / 4

    result = ppc_von_mises(
        waveform,
        get_sfreq(times),
        phase_lag,
        kappa=kappa,
        fmin=8,
        fmax=12,
        random_state=1234,
    )

    waveform = hilbert(waveform)
    result = hilbert(result)

    cplv = compute_plv(waveform, result, m=1, n=1, plv_type="complex")
    plv = np.abs(cplv)
    test_angle = np.abs(np.angle(cplv))

    # NOTE: lower and upper bounds were selected by simulating multiple time series,
    # this test should prevent large deviations from the expected result due to
    # errors in the processing
    assert lo <= plv <= hi, f"Expected PLV to be between {lo} and {hi}, got {plv}"
    if kappa >= 0.5:
        assert np.allclose(test_angle, phase_lag, atol=0.1), (
            f"Expected the actual phase lag ({test_angle:.2f}) to be within "
            f"0.1 from the desired one ({phase_lag:.2f})"
        )


@pytest.mark.parametrize("kappa,lo,hi", [(0.01, 0, 0.2), (10, 0.8, 1)])
def test_ppc_von_mises_random_envelope_kappa(kappa, lo, hi):
    _, _, times = prepare_inputs(duration=120)
    waveform = np.sin(2 * np.pi * 10 * times)
    phase_lag = np.pi / 4

    result = ppc_von_mises(
        waveform,
        get_sfreq(times),
        phase_lag,
        kappa=kappa,
        envelope="random",
        fmin=8,
        fmax=12,
        random_state=1234,
    )

    waveform = hilbert(waveform)
    result = hilbert(result)

    cplv = compute_plv(waveform, result, m=1, n=1, plv_type="complex")
    plv = np.abs(cplv)
    test_angle = np.abs(np.angle(cplv))

    # NOTE: we check only extreme cases here to make sure the random envelope does
    # not break the result completely. Still, random envelope might decrease PLV a bit,
    # so the bounds are a bit wider
    assert lo <= plv <= hi, f"Expected PLV to be between {lo} and {hi}, got {plv}"
    if kappa >= 0.5:
        assert np.allclose(np.abs(test_angle), phase_lag, atol=0.1), (
            f"Expected the actual phase lag ({test_angle:.2f}) to be within "
            f"0.1 from the desired one ({phase_lag:.2f})"
        )


@pytest.mark.parametrize("m, n", [(2, 1), (3, 1), (5 / 2, 1)])
def test_ppc_von_mises_harmonics(m, n):
    # Test with different m and n harmonics
    _, _, times = prepare_inputs()
    waveform = np.sin(2 * np.pi * 10 * times)
    phase_lag = np.pi / 4
    kappa = 10

    result = ppc_von_mises(
        waveform,
        get_sfreq(times),
        phase_lag,
        m=m,
        n=n,
        envelope="same",
        kappa=kappa,
        fmin=8,
        fmax=12,
        random_state=1234,
    )

    waveform = hilbert(waveform)
    result = hilbert(result)

    cplv = compute_plv(waveform, result, m=m, n=n, plv_type="complex")
    plv = np.abs(cplv)
    test_angle = np.angle(cplv)

    assert 0.7 <= plv, f"Expected PLV to be at least 0.7, got {plv}"
    assert np.allclose(np.abs(test_angle), phase_lag, atol=0.1), (
        f"Expected the actual phase lag ({test_angle:.2f}) to be within "
        f"0.1 from the desired one ({phase_lag:.2f})"
    )


@pytest.mark.parametrize(
    "coupling_fun,params",
    [
        (
            ppc_constant_phase_shift,
            dict(phase_lag=np.pi / 4, envelope="random", fmin=8, fmax=12),
        ),
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
    "coh,band_limited,expected_snr",
    [
        (1.0, False, np.inf),
        (1.0, True, 499.25),  # pre-computed for coh=0.999
        (1.0 / np.sqrt(2), False, 1.0),
        (1.0 / np.sqrt(2), True, 1.0),
        (0.0, False, 0.0),
        (0.0, True, 0.0),
    ],
)
def test_get_required_snr(coh, band_limited, expected_snr):
    assert np.isclose(_get_required_snr(coh, band_limited), expected_snr)


def test_get_envelope_same():
    sfreq = 250
    waveform = prepare_sinusoid(f=10, sfreq=sfreq, duration=60)
    waveform_amp = np.abs(hilbert(waveform))
    envelope = _get_envelope(waveform, envelope="same", sfreq=sfreq)
    assert np.allclose(waveform_amp, envelope), "Expected envelope to match input"


def test_get_envelope_random():
    sfreq = 250
    fmin, fmax = 8, 12
    seed = 1234
    waveform = prepare_sinusoid(f=10, sfreq=sfreq, duration=60)
    waveform_amp = np.abs(hilbert(waveform))
    envelope = _get_envelope(
        waveform,
        envelope="random",
        sfreq=sfreq,
        fmin=fmin,
        fmax=fmax,
        random_state=seed,
    )
    assert not np.allclose(
        waveform_amp, envelope
    ), "Expected envelope not to match input"


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
        band_limited=False,
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
        band_limited=False,
        random_state=seed,
    )
    actual_coh = compute_plv(waveform, coupled, m=1, n=1, coh=True)
    assert np.abs(actual_coh - target_coh) < tol


def test_shifted_copy_with_noise_infinite_snr():
    sfreq = 100
    # Multiply with the square root of 2 to normalize the variance
    waveform = np.sqrt(2) * prepare_sinusoid(f=10, sfreq=sfreq, duration=30)

    # Infinite SNR with no phase lag should return the input
    coupled = _shifted_copy_with_noise(waveform, sfreq, 0, np.inf, 8, 12, False, None)
    assert np.allclose(coupled, waveform)


@patch("meegsim.coupling.narrowband_oscillation", return_value=np.ones((100,)))
def test_shifted_copy_with_noise_zero_snr(osc_mock):
    sfreq = 100
    waveform = np.sqrt(2) * prepare_sinusoid(f=10, sfreq=sfreq, duration=30)

    # Zero SNR should return noise waveform (mock in our case)
    coupled = _shifted_copy_with_noise(waveform, sfreq, 0, 0, 8, 12, False, None)
    assert np.allclose(coupled, 1.0)
    osc_mock.assert_called_once()
