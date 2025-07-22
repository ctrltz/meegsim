"""
Tests for ensuring that subject-specific info is processed and set correctly.
"""

import mne
import pytest

from mne.datasets import sample

from meegsim.location import select_random
from meegsim.simulate import SourceSimulator
from meegsim.waveform import narrowband_oscillation

from utils.misc import running_on_ci


def prepare_real_data():
    data_path = sample.data_path() / "MEG" / "sample"
    fwd_path = data_path / "sample_audvis-meg-eeg-oct-6-fwd.fif"
    raw_path = data_path / "sample_audvis_raw.fif"
    subjects_dir = sample.data_path() / "subjects"

    # Load the prerequisites: fwd, src, and info
    fwd = mne.read_forward_solution(fwd_path)
    fwd = mne.convert_forward_solution(fwd, force_fixed=True)
    raw = mne.io.read_raw(raw_path)
    info = raw.info

    # Pick EEG channels only
    eeg_idx = mne.pick_types(info, eeg=True)
    info_eeg = mne.pick_info(info, eeg_idx)
    fwd_eeg = fwd.pick_channels(info_eeg.ch_names)

    return fwd_eeg, info_eeg, subjects_dir


@pytest.mark.skipif(running_on_ci(), reason="Skip tests with real data on CI")
def test_grow_patch_source():
    fwd, info, subjects_dir = prepare_real_data()
    src = fwd["src"]

    sfreq = 250
    duration = 60
    seed = 123

    sim = SourceSimulator(src, snr_mode="local")

    # Select some vertices randomly
    sim.add_patch_sources(
        location=select_random,
        waveform=narrowband_oscillation,
        location_params=dict(n=3),
        waveform_params=dict(fmin=8, fmax=12),
        extents=5,
        subjects_dir=subjects_dir,
        names=["s1", "s2", "s3"],
    )

    sc = sim.simulate(sfreq, duration, fwd=fwd, random_state=seed)
    sc.to_raw(fwd, info)
