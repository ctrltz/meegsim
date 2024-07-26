"""
Testing the configuration structure with white noise
"""

import numpy as np
import mne

from pathlib import Path

from meegsim.configuration import SourceConfiguration
from meegsim.waveform import white_noise

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

sc = SourceConfiguration(src, sfreq, duration, random_state=0)

# Select some vertices randomly (signal/noise does not matter for now)
source_vertno = np.random.choice(src[0]['vertno'], size=10, replace=False)
noise_vertno = np.random.choice(src[1]['vertno'], size=10, replace=False)

sc.add_point_sources([source_vertno, []], white_noise)
sc.add_noise_sources([[], noise_vertno], white_noise)

raw, stc = sc.simulate_raw(fwd, info, return_stc=True)
raw.compute_psd().plot()
input()