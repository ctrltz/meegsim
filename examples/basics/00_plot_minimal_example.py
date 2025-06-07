"""
===============
Minimal example
===============

Below you can find an example script that contains all the ideas showcased
in the Getting Started tutorial. It may serve as a good starting point
for your own simulation.
"""

import numpy as np
import mne

from mne.datasets import sample

from meegsim.coupling import ppc_von_mises
from meegsim.location import select_random
from meegsim.simulate import SourceSimulator
from meegsim.waveform import narrowband_oscillation

# %%
# First, we need to load all the prerequisites for our simulation:
#
#  * ``src`` - the :class:`mne.SourceSpaces` object that describes all candidate
#    source locations
#  * ``fwd`` - the :class:`mne.Forward` object that describes the forward model
#  * ``info`` - the :class:`mne.Info` object that describes the channel layout
#
# In this simulation, we only use the EEG channels.

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
# Below we define the parameters of the simulation and the simulation itself.
# In this case, we place 50 noise (1/f) sources randomly and add two phase-coupled
# sources of alpha (8-12 Hz) activity in arbitrary locations:

# Simulation parameters
sfreq = 100  # in Hz
duration = 60  # in seconds

# Initialize
sim = SourceSimulator(src)

# Add 50 noise sources with random locations
sim.add_noise_sources(location=select_random, location_params=dict(n=50))

# Add two point sources with fixed locations, vertex indices are chosen
# arbitrarily to have one source in each hemisphere
lh_vertno = src[0]["vertno"][0]
rh_vertno = src[1]["vertno"][0]
sim.add_point_sources(
    location=[(0, lh_vertno), (1, rh_vertno)],
    waveform=narrowband_oscillation,
    waveform_params=dict(fmin=8, fmax=12),
    names=["s1", "s2"],
)

# Set the coupling between point sources
sim.set_coupling(
    ("s1", "s2"), method=ppc_von_mises, kappa=1, phase_lag=np.pi / 2, fmin=8, fmax=12
)

# %%
# Now let's simulate the configuration with a desired level of global SNR:

sc = sim.simulate(
    sfreq,
    duration,
    fwd=fwd_eeg,
    snr_global=3,
    snr_params=dict(fmin=8, fmax=12),
    random_state=0,
)

# %%
# We can double-check that the defined sources were successfully added
# (for noise sources, we only print the total count):

print(f"Point sources: {sc._sources}")
print(f"The number of noise sources: {len(sc._noise_sources)}")

# %%
# Finally, we can obtain the simulated source activity and/or project it to
# sensor space while adding 1% of sensor noise:

stc = sc.to_stc()
raw = sc.to_raw(fwd_eeg, info_eeg, sensor_noise_level=0.01)

# %%
# We can now plot the power spectra of simulated sensor-space data to verify
# that it has a mixture of 1/f and alpha activity as defined in the simulation:

spec = raw.compute_psd(
    method="welch", n_fft=2 * sfreq, n_overlap=sfreq, n_per_seg=2 * sfreq
)
spec.plot()
