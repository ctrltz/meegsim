import numpy as np
import mne
from unittest.mock import patch

import pytest

from meegsim.snr import get_sensor_space_variance, adjust_snr


def create_dummy_sourcespace(vertices):
    # Fill in dummy data as a constant time series equal to the vertex number
    n_src_spaces = len(vertices)
    type_src = 'surf' if n_src_spaces == 2 else 'vol'
    src = []
    for i in range(n_src_spaces):
        # Create a simple dummy data structure
        n_verts = len(vertices[i])
        vertno = vertices[i]  # Vertices for this hemisphere
        xyz = np.random.rand(n_verts, 3) * 100  # Random positions
        src_dict = dict(
            vertno=vertno,
            rr=xyz,
            nn=np.random.rand(n_verts, 3),  # Random normals
            inuse=np.ones(n_verts, dtype=int),  # All vertices in use
            nuse=n_verts,
            type=type_src,
            id=i,
            np=n_verts
        )
        src.append(src_dict)

    return mne.SourceSpaces(src)


def create_dummy_forward():
    # Define the basic parameters for the forward solution
    n_sources = 10  # Number of ipoles
    n_channels = 5  # Number of MEG/EEG channels

    # Create a dummy info structure
    info = mne.create_info(ch_names=['MEG1', 'MEG2', 'MEG3', 'EEG1', 'EEG2'],
                           sfreq=1000., ch_types=['mag', 'mag', 'mag', 'eeg', 'eeg'])

    # Generate random source space data (e.g., forward operator)
    fwd_data = np.random.randn(n_channels, n_sources)

    # Create a dummy source space
    lh_vertno = np.arange(n_sources // 2)
    rh_vertno = np.arange(n_sources // 2)

    src = create_dummy_sourcespace([lh_vertno, rh_vertno])

    # Generate random source positions
    source_rr = np.random.rand(n_sources, 3)

    # Generate random source orientations
    source_nn = np.random.randn(n_sources, 3)
    source_nn /= np.linalg.norm(source_nn, axis=1, keepdims=True)

    # Create a forward solution
    forward = {
        'sol': {'data': fwd_data},
        '_orig_sol': fwd_data,
        'sol_grad': None,
        'info': info,
        'source_ori': 1,
        'surf_ori': True,
        'nsource': n_sources,
        'nchan': n_channels,
        'coord_frame': 1,
        'src': src,
        'source_rr': source_rr,
        'source_nn': source_nn,
        '_orig_source_ori': 1
    }

    # Convert the dictionary to an mne.Forward object
    fwd = mne.Forward(**forward)

    return fwd


def prepare_stc(vertices, num_samples=500):
    # Fill in dummy data as a constant time series equal to the vertex number
    data = np.tile(vertices[0] + vertices[1], reps=(num_samples, 1)).T
    return mne.SourceEstimate(data, vertices, tmin=0, tstep=0.01)


def test_get_sensor_space_variance_no_filter_all_vert():
    fwd = create_dummy_forward()
    vertices = [list(np.arange(5)), list(np.arange(5))]
    stc = prepare_stc(vertices)
    variance = get_sensor_space_variance(stc, fwd, filter=False)
    assert variance >= 0, "Variance should be non-negative"


def test_get_sensor_space_variance_no_filter_sel_vert():
    fwd = create_dummy_forward()
    vertices = [[0, 1], [0, 1]]
    stc = prepare_stc(vertices)
    variance = get_sensor_space_variance(stc, fwd, filter=False)
    assert variance >= 0, "Variance should be non-negative"


@patch('meegsim.snr.filtfilt', return_value=np.ones((1, 100)))
@patch('meegsim.snr.butter', return_value=(0, 0))
def test_get_sensor_space_variance_with_filter(butter_mock, filtfilt_mock):
    fwd = create_dummy_forward()
    vertices = [[0, 1], [0, 1]]
    stc = prepare_stc(vertices)
    variance = get_sensor_space_variance(stc, fwd, filter=True)

    # Check that butter and filtfilt were called
    butter_mock.assert_called()
    filtfilt_mock.assert_called()

    # Check that fmin and fmax are set to default values
    butter_args = butter_mock.call_args
    sfreq = stc.sfreq
    normalized_fmin = 8.0 / (0.5 * sfreq)
    normalized_fmax = 12.0 / (0.5 * sfreq)
    assert np.isclose(butter_args[0][1][0], normalized_fmin), f"Expected fmin to be {normalized_fmin}"
    assert np.isclose(butter_args[0][1][1], normalized_fmax), f"Expected fmax to be {normalized_fmax}"

    assert variance >= 0, "Variance should be non-negative"


@patch('meegsim.snr.filtfilt', return_value=np.ones((1, 100)))
@patch('meegsim.snr.butter', return_value=(0, 0))
def test_get_sensor_space_variance_with_filter_fmin_fmax(butter_mock, filtfilt_mock):
    fwd = create_dummy_forward()
    vertices = [[0, 1], [0, 1]]
    stc = prepare_stc(vertices)
    variance = get_sensor_space_variance(stc, fwd, filter=True, fmin=20., fmax=30.)

    # Check that butter and filtfilt were called
    butter_mock.assert_called()
    filtfilt_mock.assert_called()

    # Check that fmin and fmax
    butter_args = butter_mock.call_args
    sfreq = stc.sfreq
    normalized_fmin = 20.0 / (0.5 * sfreq)
    normalized_fmax = 30.0 / (0.5 * sfreq)
    assert np.isclose(butter_args[0][1][0], normalized_fmin), f"Expected fmin to be {normalized_fmin}"
    assert np.isclose(butter_args[0][1][1], normalized_fmax), f"Expected fmax to be {normalized_fmax}"

    assert variance >= 0, "Variance should be non-negative"


@pytest.mark.parametrize("target_snr", np.logspace(-6, 6, 10))
def test_adjust_snr(target_snr):
    signal_var = 10.0
    noise_var = 5.0

    snr_current = signal_var / noise_var
    expected_result = np.sqrt(snr_current / target_snr)

    result = adjust_snr(signal_var, noise_var, target_snr=target_snr)
    assert np.isclose(result, expected_result), f"Expected {expected_result}, but got {result}"


def test_adjust_snr_default_target():
    signal_var = 10.0
    noise_var = 5.0

    default_snr = 1.0
    snr_current = signal_var / noise_var
    expected_result = np.sqrt(snr_current / default_snr)

    result = adjust_snr(signal_var, noise_var)
    assert np.isclose(result, expected_result), f"Expected {expected_result}, but got {result}"


def test_adjust_snr_zero_signal_var():
    signal_var = 0.0
    noise_var = 5.0

    result = adjust_snr(signal_var, noise_var)
    assert np.isclose(result, 0.0), f"Expected 0.0, but got {result}"


def test_adjust_snr_zero_noise_var():
    signal_var = 10.0
    noise_var = 0.0

    with pytest.raises(ValueError, match="Noise variance is zero; SNR cannot be calculated."):
        adjust_snr(signal_var, noise_var)
