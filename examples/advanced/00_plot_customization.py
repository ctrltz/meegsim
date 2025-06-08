"""
Using functions from other packages
-----------------------------------

This example show how to adapt functions from other packages to
use them with MEEGsim.
"""

import matplotlib.pyplot as plt
import mne
import numpy as np

from neurodsp.sim import sim_bursty_oscillation
from neurodsp.sim.multi import sim_multiple
from mne.datasets import sample

from meegsim.location import select_random
from meegsim.simulate import SourceSimulator
from meegsim.utils import normalize_variance

# %%
# For this example, we use the :meth:`~neurodsp.sim.sim_bursty_oscillation` function
# from the `NeuroDSP <https://neurodsp-tools.github.io/neurodsp/>`_ package. To make
# the function compatible with MEEGsim, we need to wrap it in another function and
# adapt the input and output parameters:


def bursty_osc(n_series, times, **kwargs):
    # Convert MEEGsim input to NeuroDSP input
    tstep = times[1] - times[0]
    n_seconds = times.max() + tstep
    fs = 1.0 / tstep

    params = dict(n_seconds=n_seconds, fs=fs)
    params.update(kwargs)

    # Random state is not accepted by NeuroDSP function, use it to set
    # random starting phase for some variability
    seed = params.pop("random_state")
    phase = np.random.default_rng(seed).random()
    params["phase"] = phase

    sims = sim_multiple(sim_bursty_oscillation, params, n_sims=n_series)

    return normalize_variance(sims.signals)


# %%
# Load the source space and a head model for EEG channels:

# Paths
data_path = sample.data_path() / "MEG" / "sample"
fwd_path = data_path / "sample_audvis-meg-eeg-oct-6-fwd.fif"
raw_path = data_path / "sample_audvis_raw.fif"

# Load the prerequisites: fwd, src, and info
fwd = mne.read_forward_solution(fwd_path)
raw = mne.io.read_raw(raw_path)
src = fwd["src"]
info = raw.info

# Pick EEG channels only
eeg_idx = mne.pick_types(info, eeg=True)
info_eeg = mne.pick_info(info, eeg_idx)
fwd_eeg = fwd.pick_channels(info_eeg.ch_names)

# %%
# Now we can use the adapted function in a simulation, providing the required
# parameters via the ``waveform_params`` dictionary:

sim = SourceSimulator(src)

sim.add_point_sources(
    location=select_random,
    location_params=dict(n=2),
    waveform=bursty_osc,
    waveform_params=dict(  # NeuroDSP parameters
        freq=20,
        burst_def="durations",
        burst_params={"n_cycles_burst": 3, "n_cycles_off": 3},
    ),
    names=["lh", "rh"],
)

sc = sim.simulate(sfreq=250, duration=60, fwd=fwd, random_state=123)

# %%
# We can check that the simulation of a bursty oscillation actually worked:

n_samples_to_plot = 500
fig, ax = plt.subplots()
ax.plot(sc.times[:n_samples_to_plot], sc._sources["lh"].waveform[:n_samples_to_plot])
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude (nAm)")

# %%
#
# .. note::
#    Coupling methods provided by MEEGsim might not preserve specific properties
#    of time courses generated via functions from other packages. (e.g., presence
#    of bursts in this example).
