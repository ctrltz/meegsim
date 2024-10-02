"""
Testing the configuration structure
"""

import json
import mne
import numpy as np

from harmoni.extratools import compute_plv
from pathlib import Path

from meegsim.coupling import ppc_von_mises
from meegsim.location import select_random
from meegsim.simulate import SourceSimulator
from meegsim.waveform import narrowband_oscillation


def to_json(sources):
    return json.dumps({k: str(s) for k, s in sources.items()}, indent=4)


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
target_snr = 20

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
    names=['s1', 's2', 's3']
)

# Set coupling
sim.set_coupling(coupling={
    ('s1', 's2'): dict(kappa=1, phase_lag=np.pi/3),
    ('s2', 's3'): dict(kappa=0.5, phase_lag=-np.pi/6)
}, method=ppc_von_mises, fmin=8, fmax=12)

print(sim._coupling)

sc = sim.simulate(sfreq, duration, fwd=fwd, random_state=seed)
stc = sc.to_stc()
raw = sc.to_raw(fwd, info)

print('PLV:', compute_plv(stc.data, stc.data, n=1, m=1))
print('iPLV:', compute_plv(stc.data, stc.data, n=1, m=1, plv_type='imag'))

spec = raw.compute_psd(n_fft=sfreq, n_overlap=sfreq//2, n_per_seg=sfreq)
spec.plot(sphere='eeglab')
input('Press any key to continue')
