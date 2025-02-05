"""
Testing the configuration structure
"""

import json
import matplotlib.pyplot as plt
import mne
import numpy as np

from pathlib import Path

from meegsim.location import select_random
from meegsim.simulate import SourceSimulator
from meegsim.waveform import narrowband_oscillation


def to_json(sources):
    return json.dumps({k: str(s) for k, s in sources.items()}, indent=4)


def data2stc(data, src):
    vertno = [s["vertno"] for s in src]
    return mne.SourceEstimate(
        data=data, vertices=vertno, tmin=0, tstep=0.01, subject="fsaverage"
    )


def extents_from_areas_cm2(areas_cm2):
    return list(np.sqrt(np.array(areas_cm2) * 100 / np.pi))


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

# Create a dummy stc for std based on the y-position of the sources
ypos = np.hstack([1 - 8 * np.abs(s["rr"][s["inuse"] > 0, 1]) for s in src])
std_stc = data2stc(ypos, src)
std_stc.plot(
    subject="fsaverage",
    hemi="split",
    views=["lat", "med"],
    clim=dict(kind="value", lims=[0, 1, 2]),
    transparent=False,
    background="white",
)

sim = SourceSimulator(src)

sim.add_noise_sources(location=select_random, location_params=dict(n=10))

# Select some vertices randomly
sim.add_patch_sources(
    location=select_random,
    waveform=narrowband_oscillation,
    location_params=dict(n=3),
    waveform_params=dict(fmin=8, fmax=12),
    std=std_stc,
    extents=extents_from_areas_cm2([2, 4, 8]),
)

sc = sim.simulate(
    sfreq,
    duration,
    fwd=fwd,
    snr_global=4,
    snr_params=dict(fmin=8, fmax=12),
    random_state=seed,
)
stc = sc.to_stc()
raw = sc.to_raw(fwd, info, sensor_noise_level=0.05)

source_std = np.std(stc.data, axis=1)
lim = np.max(source_std)
std_stc_est = mne.SourceEstimate(source_std, stc.vertices, tmin=0, tstep=0.01)
std_stc_est.plot(
    subject="fsaverage",
    hemi="split",
    views=["lat", "med"],
    clim=dict(kind="value", lims=[0, lim / 2, lim]),
    colormap="Reds",
    time_viewer=False,
    transparent=False,
    background="white",
)

sc.plot(subject="fsaverage", hemi="split", views=["lat", "med"])

spec = raw.compute_psd(n_fft=sfreq, n_overlap=sfreq // 2, n_per_seg=sfreq)
spec.plot_topomap(bands={"alpha": (8, 12)}, sphere="eeglab")
spec.plot(sphere="eeglab")
plt.show(block=True)
