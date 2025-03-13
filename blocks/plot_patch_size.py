"""
Cortical patches
================

This example shows how to control the area of the patch and highlights its
effect on the resulting leadfield of the source.
"""

import mne

from mne.datasets import sample

data_path = sample.data_path()
meg_path = data_path / "MEG" / "sample"
fwd_path = meg_path / "sample_audvis-meg-eeg-oct-6-fwd.fif"

fwd = mne.read_forward_solution(fwd_path)
indices_by_type = mne.channel_indices_by_type(fwd["info"])
eeg_ch_names = fwd.ch_names[indices_by_type["eeg"]]
fwd_eeg = fwd.pick_channels(eeg_ch_names)
