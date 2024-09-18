"""
Testing the configuration structure
"""

import json
import mne
import numpy as np

from pathlib import Path
from scipy.signal import butter, filtfilt

from meegsim.location import select_random
from meegsim.simulate import SourceSimulator
from meegsim.waveform import narrowband_oscillation


def to_json(sources):
    return json.dumps({k: str(s) for k, s in sources.items()}, indent=4)


# Load the head model
fs_dir = Path(mne.datasets.fetch_fsaverage('~/mne_data/MNE-fsaverage-data'))
fwd_path = fs_dir / 'bem_copy' / 'fsaverage-oct6-fwd.fif'
src_path = fs_dir / 'bem_copy' / 'fsaverage-oct6-src.fif'
src = mne.read_source_spaces(src_path)
fwd = mne.read_forward_solution(fwd_path)

# Simulation parameters
sfreq = 250
duration = 60
seed = 1234
target_snr = 20

b, a = butter(4, 2 * np.array([8, 12]) / sfreq, 'bandpass')

# Channel info
montage = mne.channels.make_standard_montage('standard_1020')
ch_names = [ch for ch in montage.ch_names if ch not in ['O9', 'O10']]
info = mne.create_info(ch_names, sfreq, ch_types='eeg')
info.set_montage('standard_1020')

# Adapt fwd to the info (could be done by our structure in principle)
fwd = mne.convert_forward_solution(fwd, force_fixed=True)
fwd = mne.pick_channels_forward(fwd, info.ch_names, ordered=True)

sim = SourceSimulator(src)

# Add noise sources
sim.add_noise_sources(
    location=select_random,
    location_params=dict(n=10)
)

sc_noise = sim.simulate(sfreq, duration, random_state=seed)
raw_noise = sc_noise.to_raw(fwd, info)

# Select some vertices randomly
sim.add_point_sources(
    location=select_random,
    waveform=narrowband_oscillation,
    location_params=dict(n=1),
    waveform_params=dict(fmin=8, fmax=12),
    snr=target_snr,
    snr_params=dict(fmin=8, fmax=12)
)

sc_full = sim.simulate(sfreq, duration, fwd=fwd, random_state=seed)
raw_full = sc_full.to_raw(fwd, info)

n_samples = sc_full.times.size
noise_data = filtfilt(b, a, raw_noise.get_data())
cov_raw_noise = (noise_data @ noise_data.T) / n_samples
full_data = filtfilt(b, a, raw_full.get_data())
cov_raw_full = (full_data @ full_data.T) / n_samples
snr = np.mean(np.diag(cov_raw_full)) / np.mean(np.diag(cov_raw_noise)) - 1
print(f'Target SNR = {target_snr:.2f}')
print(f'Actual SNR = {snr:.2f}')

spec = raw_full.compute_psd(n_fft=sfreq, n_overlap=sfreq//2, n_per_seg=sfreq)
spec.plot(sphere='eeglab')
input('Press any key to continue')