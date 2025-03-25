import matplotlib.pyplot as plt
import mne
import numpy as np

from matplotlib.gridspec import GridSpec
from pathlib import Path

from meegsim.coupling import ppc_shifted_copy_with_noise
from meegsim.location import select_random
from meegsim.waveform import narrowband_oscillation
from meegsim.simulate import SourceSimulator


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
n_sources = 3

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

# Select some vertices randomly
for i in range(3):
    sim.add_point_sources(
        location=select_random,
        waveform=narrowband_oscillation,
        location_params=dict(n=1),
        waveform_params=dict(fmin=8, fmax=12),
        names=[str(i)],
    )
sim.set_coupling(
    ("0", "1"),
    method=ppc_shifted_copy_with_noise,
    phase_lag=np.pi / 2,
    coh=0.5,
    fmin=8,
    fmax=12,
)


sc = sim.simulate(sfreq=sfreq, duration=duration, random_state=1234)
brain = sc.plot(subject="fsaverage", hemi="split", views=["lat", "med"])
screenshot = brain.screenshot()
brain.close()

print([s.vertno for s in sc._sources.values()])

fig = plt.figure(figsize=(12, 6))
gs = GridSpec(3, 2, figure=fig)
ax_brain = fig.add_subplot(gs[:, 0])
ax_brain.imshow(screenshot)
ax_brain.axis("off")
for i in range(3):
    s = sc._sources[str(i)]
    ax = fig.add_subplot(gs[i, 1])
    ax.plot(sc.times[:1000], np.squeeze(s.waveform)[:1000])
fig.tight_layout()
