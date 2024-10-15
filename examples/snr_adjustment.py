"""
Testing the adjustment of SNR
"""

import mne
import numpy as np
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

fig, axes = plt.subplots(ncols=3, figsize=(8, 3))
snr_values = [1, 5, 10]

for i_snr, target_snr in enumerate(snr_values):
    sim = SourceSimulator(src)

    # Select some vertices randomly
    sim.add_point_sources(
        location=select_random,
        waveform=narrowband_oscillation,
        location_params=dict(n=3),
        waveform_params=dict(fmin=8, fmax=12),
        snr=target_snr,
        snr_params=dict(fmin=8, fmax=12),
        names=['s1', 's2', 's3']
    )

    sim.add_noise_sources(
        location=select_random,
        location_params=dict(n=10)
    )

    sc = sim.simulate(sfreq, duration, fwd=fwd, random_state=seed)
    raw = sc.to_raw(fwd, info)

    spec = raw.compute_psd(fmax=40, n_fft=sfreq, 
                           n_overlap=sfreq//2, n_per_seg=sfreq)
    spec.plot(average=True, dB=False, axes=axes[i_snr], amplitude=False)
    
    axes[i_snr].set_title(f'SNR={target_snr}')
    axes[i_snr].set_xlabel('Frequency (Hz)')
    axes[i_snr].set_ylabel('PSD (uV^2/Hz)')
    axes[i_snr].set_ylim([0, 1.25])

fig.tight_layout()
plt.show(block=True)
