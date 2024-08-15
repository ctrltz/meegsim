"""
Testing the configuration structure
"""

import json
import numpy as np
import mne

from pathlib import Path

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

# Channel info
montage = mne.channels.make_standard_montage('standard_1020')
ch_names = [ch for ch in montage.ch_names if ch not in ['O9', 'O10']]
info = mne.create_info(ch_names, sfreq, ch_types='eeg')
info.set_montage('standard_1020')

# Adapt fwd to the info (could be done by our structure in principle)
fwd = mne.convert_forward_solution(fwd, force_fixed=True)
fwd = mne.pick_channels_forward(fwd, info.ch_names, ordered=True)

sim = SourceSimulator(src)

# Select some vertices randomly (signal/noise does not matter for now)
sim.add_point_sources(
    location=select_random, 
    waveform=narrowband_oscillation,
    location_params=dict(n=10, vertices=[list(src[0]['vertno']), []]),
    waveform_params=dict(fmin=8, fmax=12)
)
sim.add_noise_sources(
    location=select_random,
    location_params=dict(n=10, vertices=[[], list(src[1]['vertno'])])
)

# Print the source groups to check internal structure
print(sim._source_groups)
print(sim._noise_groups)

sc = sim.simulate(sfreq, duration, random_state=0)
raw = sc.to_raw(fwd, info)

# Print the sources to check internal structure
print(to_json(sc._sources))

raw, stc = sc.simulate_raw(fwd, info, return_stc=True)
spec = raw.compute_psd(n_fft=sfreq, n_overlap=sfreq//2, n_per_seg=sfreq)
spec.plot(sphere='eeglab')
input('Press any key to continue')