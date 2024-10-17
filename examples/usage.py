"""
Planned usage - as a first milestone, it would be nice to get this script to work.
"""

import numpy as np

from functools import partial

from meegsim import SourceConfiguration
from meegsim.location import select_random, select_random_in_labels
from meegsim.waveform import narrowband_oscillation


# Users can create such shortcuts if needed
alpha_oscillation = partial(narrowband_oscillation,
                            fmin=8, fmax=12, order=2)

sc = SourceConfiguration(src, sfreq, duration, random_state=0)

# Point sources of alpha
alpha_vertices = select_random_in_labels(src, labels)
sc.add_point_sources(alpha_vertices, waveform=alpha_oscillation, snr=1,
                    names=['m1-left', 's1-left', 'm1-right', 's1-right'])

# Point sources with 1/f noise, separate function allows marking these sources as noise for SNR calculations					       
sc.add_noise_sources(select_random, location_params=dict(n=1000))

# TODO: add patch sources
sc.add_patch_sources()

# Coupling is a dictionary with connectivity edges as keys and coupling parameters as values
coupling = {
    ('m1-left', 's1-left'): dict(kappa=1, dphi=np.pi/2),
    ('m1-left', 'm1-right'): dict(kappa=0.5, dphi=np.pi/4)
}
sc.set_coupling(coupling, coupling_fun='ppc_von_mises')

# Project to sensor space, optionally get corresponding stc
raw, stc = sc.simulate_raw(fwd, return_stc=True)

# Quick access of ground truth waveforms for sources of interest
gt = sc.get_waveforms(['m1-left', 'm1-right'])