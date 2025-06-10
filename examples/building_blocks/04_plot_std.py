"""
Controlling the standard deviation of activity
==============================================

"""

import mne
import numpy as np

from mne.datasets import sample

from meegsim.location import select_random
from meegsim.simulate import SourceSimulator
from meegsim.waveform import narrowband_oscillation


def data2stc(data, src):
    vertno = [s["vertno"] for s in src]
    return mne.SourceEstimate(
        data=data, vertices=vertno, tmin=0, tstep=0.01, subject="fsaverage"
    )


def extents_from_areas_cm2(areas_cm2):
    return list(np.sqrt(np.array(areas_cm2) * 100 / np.pi))


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
target_snr = 20

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
    subject="sample",
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
