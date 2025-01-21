"""
Testing the sensor space noise
"""

import mne
import matplotlib.pyplot as plt

from pathlib import Path

from meegsim.location import select_random
from meegsim.simulate import SourceSimulator
from meegsim.waveform import narrowband_oscillation


# Load the head model
fs_dir = Path('~/mne_data/MNE-fsaverage-data/fsaverage/')
fwd_path = fs_dir / 'bem_copy' / 'fsaverage-oct6-fwd.fif'
src_path = fs_dir / 'bem_copy' / 'fsaverage-oct6-src.fif'
src = mne.read_source_spaces(src_path)
fwd = mne.read_forward_solution(fwd_path)

# Simulation parameters
sfreq = 250
duration = 60
seed = 123

# Channel info
montage = mne.channels.make_standard_montage('standard_1020')
ch_names = [ch for ch in montage.ch_names if ch not in ['O9', 'O10']]
info = mne.create_info(ch_names, sfreq, ch_types='eeg')
info.set_montage('standard_1020')

# Adapt fwd to the info (could be done by our structure in principle)
fwd = mne.convert_forward_solution(fwd, force_fixed=True)
fwd = mne.pick_channels_forward(fwd, info.ch_names, ordered=True)

sim = SourceSimulator(src)

# Select some vertices randomly
sim.add_point_sources(
    location=select_random,
    waveform=narrowband_oscillation,
    location_params=dict(n=3),
    waveform_params=dict(fmin=8, fmax=12),
    snr=2.5,
    snr_params=dict(fmin=8, fmax=12),
    names=['s1', 's2', 's3']
)

sim.add_noise_sources(
    location=select_random,
    location_params=dict(n=10)
)

sc = sim.simulate(sfreq, duration, fwd=fwd, random_state=seed)
raw = sc.to_raw(fwd, info)

noise_levels = [0, 0.05, 0.1, 0.25, 0.5, 1]
n_levels = len(noise_levels)
fig, axes = plt.subplots(ncols=n_levels, figsize=(3 * n_levels, 3))

for i_level, noise_level in enumerate(noise_levels):
    raw = sc.to_raw(fwd, info, sensor_noise_level=noise_level)

    spec = raw.compute_psd(fmax=60, n_fft=sfreq, 
                           n_overlap=sfreq//2, n_per_seg=sfreq)
    spec.plot(axes=axes[i_level], amplitude=False, sphere='eeglab')
    
    axes[i_level].set_title(f'{noise_level=}')
    axes[i_level].set_xlabel('Frequency (Hz)')

fig.tight_layout()
plt.show(block=True)
