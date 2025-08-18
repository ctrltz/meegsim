"""
Adding sensor space noise
=========================

"""

import mne
import matplotlib.pyplot as plt

from mne.datasets import sample

from meegsim.location import select_random
from meegsim.simulate import SourceSimulator
from meegsim.waveform import narrowband_oscillation

# %%
# First, we load the head model and associated source space:

# Paths
data_path = sample.data_path() / "MEG" / "sample"
fwd_path = data_path / "sample_audvis-meg-eeg-oct-6-fwd.fif"
raw_path = data_path / "sample_audvis_raw.fif"

# Load the prerequisites: fwd, src, and info
fwd = mne.read_forward_solution(fwd_path)
fwd = mne.convert_forward_solution(fwd, force_fixed=True)
raw = mne.io.read_raw(raw_path)
src = fwd["src"]
info = raw.info

# Pick EEG channels only
eeg_idx = mne.pick_types(info, eeg=True)
info_eeg = mne.pick_info(info, eeg_idx)
fwd_eeg = fwd.pick_channels(info_eeg.ch_names)

# Simulation parameters
sfreq = 250
duration = 60
seed = 123

sim = SourceSimulator(src)

# Select some vertices randomly
sim.add_point_sources(
    location=select_random,
    waveform=narrowband_oscillation,
    location_params=dict(n=3),
    waveform_params=dict(fmin=8, fmax=12),
    snr=2.5,
    snr_params=dict(fmin=8, fmax=12),
    names=["s1", "s2", "s3"],
)

sim.add_noise_sources(location=select_random, location_params=dict(n=10))

sc = sim.simulate(sfreq, duration, fwd=fwd, random_state=seed)
raw = sc.to_raw(fwd, info)

noise_levels = [0.05, 0.25, 0.5]
n_levels = len(noise_levels)
fig, axes = plt.subplots(ncols=n_levels, figsize=(3 * n_levels, 3))

for i_level, noise_level in enumerate(noise_levels):
    raw = sc.to_raw(fwd, info, sensor_noise_level=noise_level)

    spec = raw.compute_psd(fmax=60, n_fft=sfreq, n_overlap=sfreq // 2, n_per_seg=sfreq)
    spec.plot(axes=axes[i_level], amplitude=False)

    axes[i_level].set_title(f"{noise_level=}")
    axes[i_level].set_xlabel("Frequency (Hz)")

fig.tight_layout()
