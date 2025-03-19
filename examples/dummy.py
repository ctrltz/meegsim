"""
Testing the configuration structure
===================================

"""

import json
import matplotlib.pyplot as plt
import mne
import numpy as np

from pathlib import Path

from meegsim.coupling import ppc_von_mises
from meegsim.location import select_random
from meegsim.simulate import SourceSimulator
from meegsim.waveform import narrowband_oscillation


def to_json(sources):
    return json.dumps({k: str(s) for k, s in sources.items()}, indent=4)


# Load the head model
fs_dir = Path("~/mne_data/MNE-fsaverage-data/fsaverage/")
fwd_path = fs_dir / "bem_copy" / "fsaverage-oct6-fwd.fif"
src_path = fs_dir / "bem_copy" / "fsaverage-oct6-src.fif"
src = mne.read_source_spaces(src_path)
fwd = mne.read_forward_solution(fwd_path)

# Simulation parameters
sfreq = 250
duration = 60
seed = 123
target_snr = 20

# Channel info
montage = mne.channels.make_standard_montage("standard_1020")
ch_names = [
    ch for ch in montage.ch_names if ch not in ["O9", "O10", "T3", "T4", "T5", "T6"]
]
info = mne.create_info(ch_names, sfreq, ch_types="eeg")
info.set_montage("standard_1020")

# Adapt fwd to the info (could be done by our structure in principle)
fwd = mne.convert_forward_solution(fwd, force_fixed=True)
fwd = mne.pick_channels_forward(fwd, info.ch_names, ordered=True)

sim = SourceSimulator(src)

sim.add_noise_sources(location=select_random, location_params=dict(n=10))

# Select some vertices randomly
sim.add_point_sources(
    location=[(0, 0), (0, 87780), (0, 106307)],
    waveform=narrowband_oscillation,
    location_params=dict(n=3),
    waveform_params=dict(fmin=8, fmax=12),
    std=[1, 1, 10],
    names=["s1", "s2", "s3"],
)

sim.add_patch_sources(
    location=select_random,
    waveform=narrowband_oscillation,
    snr=1,
    location_params=dict(n=3),
    waveform_params=dict(fmin=8, fmax=12),
    snr_params=dict(fmin=8, fmax=12),
    extents=[10, 20, 50],
    names=["s4", "s5", "s6"],
)

# Set coupling
sim.set_coupling(
    coupling={
        ("s1", "s2"): dict(kappa=1, phase_lag=np.pi / 3),
        ("s2", "s3"): dict(kappa=10, phase_lag=-np.pi / 2),
    },
    method=ppc_von_mises,
    fmin=8,
    fmax=12,
)

print(sim._coupling_graph)
print(sim._coupling_graph.edges(data=True))

sc = sim.simulate(
    sfreq,
    duration,
    fwd=fwd,
    snr_global=4,
    snr_params=dict(fmin=8, fmax=12),
    random_state=seed,
)
raw = sc.to_raw(fwd, info, sensor_noise_level=0.05)

print(sc._sources["s1"].to_label(src))
print(sc._sources["s4"].to_label(src))
print([np.var(s.waveform) for s in sc._sources.values()])

sc.plot(subject="fsaverage", hemi="split", views=["lat", "med"])

spec = raw.compute_psd(n_fft=sfreq, n_overlap=sfreq // 2, n_per_seg=sfreq)
spec.plot_topomap(bands={"alpha": (8, 12)}, sphere="eeglab")
spec.plot(sphere="eeglab")
plt.show(block=True)
