import numpy as np
import pytest

from mock import patch
from meegsim.configuration import SourceConfiguration
from meegsim.sources import PointSource, PatchSource

from utils.prepare import prepare_source_space, prepare_forward


def test_sourceconfiguration_to_stc_empty_raises():
    src = prepare_source_space(types=["surf", "surf"], vertices=[[0, 1], [0, 1]])

    # empty source configuration
    sc = SourceConfiguration(src, sfreq=250, duration=30)
    with pytest.raises(ValueError, match="No sources were added"):
        sc.to_stc()


def test_sourceconfiguration_to_stc_noise_only():
    src = prepare_source_space(types=["surf", "surf"], vertices=[[0, 1], [0, 1]])

    sc = SourceConfiguration(src, sfreq=250, duration=30)
    sc._noise_sources = {
        "n1": PointSource("n1", 0, 0, np.ones((250 * 30,))),
        "n2": PointSource("n2", 0, 1, np.ones((250 * 30,))),
    }
    stc = sc.to_stc()
    assert stc.data.shape[0] == 2, "Expected two sources in stc"


def test_sourceconfiguration_to_stc_signal_only():
    src = prepare_source_space(types=["surf", "surf"], vertices=[[0, 1], [0, 1]])

    sc = SourceConfiguration(src, sfreq=250, duration=30)
    sc._sources = {
        "s1": PointSource("s1", 0, 0, np.ones((250 * 30,))),
        "s2": PointSource("s2", 0, 1, np.ones((250 * 30,))),
    }
    stc = sc.to_stc()
    assert stc.data.shape[0] == 2, "Expected two sources in stc"


def test_sourceconfiguration_to_stc_signal_and_noise():
    src = prepare_source_space(types=["surf", "surf"], vertices=[[0, 1], [0, 1]])

    sc = SourceConfiguration(src, sfreq=250, duration=30)
    sc._sources = {
        "s1": PointSource("s1", 0, 0, np.ones((250 * 30,))),
    }
    sc._noise_sources = {
        "n1": PointSource("n1", 0, 1, np.ones((250 * 30,))),
    }
    stc = sc.to_stc()
    assert stc.data.shape[0] == 2, "Expected two sources in stc"


def test_sourceconfiguration_to_stc_patch():
    src = prepare_source_space(types=["surf", "surf"], vertices=[[0, 1, 2], [0, 1, 2]])

    sc = SourceConfiguration(src, sfreq=250, duration=30)
    n_samples = sc.sfreq * sc.duration
    sources = [
        PatchSource("s1", 0, [0, 2], np.ones((n_samples,))),
        PatchSource("s2", 1, [0, 1], np.ones((n_samples,))),
    ]
    sc._sources = {s.name: s for s in sources}
    stc = sc.to_stc()
    assert stc.data.shape[0] == 4, "Expected four sources in stc"


@patch("mne.apply_forward_raw", return_value=0)
def test_sourceconfiguration_to_raw(apply_forward_mock):
    src = prepare_source_space(types=["surf", "surf"], vertices=[[0, 1], [0, 1]])

    sc = SourceConfiguration(src, sfreq=250, duration=30)
    n_samples = sc.sfreq * sc.duration
    sc._sources = {
        "s1": PointSource("s1", 0, 0, np.ones((n_samples,))),
        # adjust the waveform to account for scaling in .data
        "s2": PatchSource("s2", 1, [0, 1], np.ones((n_samples,)) * np.sqrt(2)),
    }
    sc._noise_sources = {
        "n1": PointSource("n1", 0, 1, np.ones((n_samples,))),
    }

    raw = sc.to_raw([], [])
    apply_forward_mock.assert_called()
    stc = apply_forward_mock.call_args.args[1]
    assert np.allclose(stc.data, 1.0), "Source time courses should not be changed"
    assert raw == 0, "Output of apply_forward_raw should not be changed"


@patch("meegsim.configuration._adjust_sensor_noise", return_value=0)
def test_sourceconfiguration_to_raw_sensor_space_noise(adjust_noise_mock):
    src = prepare_source_space(types=["surf", "surf"], vertices=[[0, 1], [0, 1]])
    fwd = prepare_forward(5, 4)

    sc = SourceConfiguration(src, sfreq=250, duration=30)
    n_samples = sc.sfreq * sc.duration
    sc._sources = {
        "s1": PointSource("s1", 0, 0, np.ones((n_samples,))),
        "s2": PatchSource("s2", 1, [0, 1], np.ones((n_samples,))),
    }
    sc._noise_sources = {
        "n1": PointSource("n1", 0, 1, np.ones((n_samples,))),
    }

    # By default, no sensor space noise is added
    raw = sc.to_raw(fwd, fwd["info"])
    adjust_noise_mock.assert_not_called()

    # Once called, mock return value should be returned
    raw = sc.to_raw(fwd, fwd["info"], sensor_noise_level=0.1)
    adjust_noise_mock.assert_called_once()
    assert raw == 0
