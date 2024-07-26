import numpy as np

from meegsim.waveform import white_noise


def prepare_inputs():
    n_series = 10
    n_times = 100
    times = np.linspace(0, 1, num=n_times)
    return n_series, n_times, times


def test_white_noise_shape():
    n_series, n_times, times = prepare_inputs()

    data = white_noise(n_series, times)
    assert data.shape == (n_series, n_times)


def test_white_noise_random_state():
    n_series, _, times = prepare_inputs()

    # Different time series are generated by default
    data1 = white_noise(n_series, times)
    data2 = white_noise(n_series, times)
    assert not np.allclose(data1, data2)

    # The results are reproducible when random_state is set
    random_state = 1234567890
    data1 = white_noise(n_series, times, random_state=random_state)
    data2 = white_noise(n_series, times, random_state=random_state)
    assert np.allclose(data1, data2)
