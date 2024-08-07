import numpy as np
from src.meegsim.coupling import shifted_copy, ppc_von_mises
from scipy.signal import hilbert

def test_real_waveform():
    # Test with a simple sinusoidal waveform
    t = np.linspace(0, 1, 500)
    waveform = np.sin(2 * np.pi * 5 * t)
    phase_lag = np.pi / 3

    result = shifted_copy(waveform, phase_lag)[1]

    waveform = hilbert(waveform)
    result = hilbert(result)

    waveform_angle = np.angle(waveform)
    result_angle = np.angle(result)

    plv = np.abs(np.mean(np.exp(1j * waveform_angle - 1j * result_angle)))

    test_angle = waveform_angle - result_angle
    test_angle = (test_angle + np.pi) % (2 * np.pi) - np.pi

    assert abs(plv - 1) <= 0.01, "Test failed: plv is smaller than 0.99."
    assert (np.abs(np.mean(test_angle)) - phase_lag) <= 0.01, "Test failed: angle is different from phase_lag."


def test_complex_waveform():
    # Test with a complex waveform
    t = np.linspace(0, 1, 500)
    waveform = np.exp(1j * 2 * np.pi * 5 * t)
    phase_lag = np.pi / 4

    result = shifted_copy(waveform, phase_lag)[1]

    result = hilbert(result)

    waveform_angle = np.angle(waveform)
    result_angle = np.angle(result)

    plv = np.abs(np.mean(np.exp(1j * waveform_angle - 1j * result_angle)))

    test_angle = waveform_angle - result_angle
    test_angle = (test_angle + np.pi) % (2 * np.pi) - np.pi

    assert abs(plv - 1) <= 0.01, "Test failed: plv is smaller than 0.99."
    assert (np.abs(np.mean(test_angle)) - phase_lag) <= 0.01, "Test failed: angle is different from phase_lag."


def test_different_harmonics():
    #TODO
    # Make it work for harmonics other than 1, 2

    # Test with different m and n harmonics
    t = np.linspace(0, 1, 500)
    waveform = np.sin(2 * np.pi * 5 * t)  # 5 Hz sinusoid
    phase_lag = np.pi / 3
    m, n = 1, 2  # Different harmonics

    result = shifted_copy(waveform, phase_lag, m=m, n=n)[1]

    waveform = hilbert(waveform)
    result = hilbert(result)

    waveform_angle = np.angle(waveform)
    result_angle = np.angle(result)

    plv = np.abs(np.mean(np.exp(1j * waveform_angle - 1j * result_angle)))

    test_angle = waveform_angle - result_angle
    test_angle = (test_angle + np.pi) % (2 * np.pi) - np.pi

    assert abs(plv - 1) <= 0.2, "Test failed: plv is smaller than 0.8."
    assert (np.abs(np.mean(test_angle)) - phase_lag) <= 0.2, "Test failed: angle is different from phase_lag."


def test_real_waveform_vonmises():
    # Test kappas that are reliable (more than 0.5)
    fs = 1000
    t = np.linspace(0, 1, fs)
    waveform = np.sin(2 * np.pi * 10 * t)
    phase_lag = np.pi / 4
    kappa = 0.6

    result = ppc_von_mises(waveform, fs, phase_lag, kappa=kappa)[1]

    waveform = hilbert(waveform)
    result = hilbert(result)

    waveform_angle = np.angle(waveform)
    result_angle = np.angle(result)

    plv = np.abs(np.mean(np.exp(1j * waveform_angle - 1j * result_angle)))

    test_angle = waveform_angle - result_angle
    test_angle = (test_angle + np.pi) % (2 * np.pi) - np.pi

    if kappa > 0.5:
        assert abs(plv - 1) <= 0.2, "Test failed: plv is smaller than 0.9."
        assert (np.abs(np.mean(test_angle)) - phase_lag) <= 0.2, "Test failed: angle is different from phase_lag."

#TODO
# Come up with the test for small kappas

def test_reproducibility_with_random_state():
    # Test that using a fixed random state gives the same result
    fs = 1000
    t = np.linspace(0, 1, fs)
    waveform = np.sin(2 * np.pi * 10 * t)
    phase_lag = np.pi / 4
    random_state = 42

    result1 = ppc_von_mises(waveform, fs, phase_lag, random_state=random_state)[1]
    result2 = ppc_von_mises(waveform, fs, phase_lag, random_state=random_state)[1]

    # Test that results are identical
    np.testing.assert_array_almost_equal(result1, result2)