"""
Integration tests
"""

import numpy as np

from meegsim.coupling import constant_phase_shift, ppc_von_mises
from meegsim.location import select_random
from meegsim.simulate import SourceSimulator
from meegsim.waveform import narrowband_oscillation

from utils.prepare import prepare_forward


def test_builtin_methods():
    """
    In this test, we define a simulation that should involve all 
    built-in methods and main features that are provided by the toolbox.
    """

    # Define the dummy forward model
    fwd = prepare_forward(5, 10)
    src = fwd["src"]
    info = fwd["info"]

    # Simulation parameters
    sfreq = 250
    duration = 10
    seed = 1234

    sim = SourceSimulator(src)

    # 1/f noise (default), fixed location
    sim.add_noise_sources(location=[(0, 0), (0, 2), (0, 4)])

    # White noise, random location
    sim.add_noise_sources(
        location=select_random,
        location_params=dict(n=2)
    )

    # Sources of oscillatory activity with target SNR
    sim.add_point_sources(
        location=select_random,
        location_params=dict(n=3),
        waveform=narrowband_oscillation,
        waveform_params=dict(fmin=8, fmax=12),
        snr=5,
        snr_params=dict(fmin=8, fmax=12),
        names=['s1', 's2', 's3']
    )

    # Define several edges to test graph traversal and built-in coupling methods
    sim.set_coupling(('s1', 's2'), method=ppc_von_mises, kappa=1, 
                     phase_lag=np.pi/3, fmin=8, fmax=12)
    sim.set_coupling(coupling={
        ('s2', 's3'): dict(phase_lag=-np.pi/6),
    }, method=constant_phase_shift, phase_lag=0)

    # Actual simulation
    sc = sim.simulate(sfreq, duration, fwd, random_state=seed)

    # SourceConfiguration methods
    sc.to_stc()
    sc.to_raw(fwd, info)
