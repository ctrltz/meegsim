"""
Integration tests
"""

import numpy as np

from meegsim.coupling import ppc_constant_phase_shift, ppc_von_mises
from meegsim.location import select_random
from meegsim.simulate import SourceSimulator
from meegsim.waveform import narrowband_oscillation, white_noise

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

    sim = SourceSimulator(src, snr_mode="local")

    # 1/f noise (default), fixed location
    sim.add_noise_sources(location=[(0, 0), (0, 2), (0, 4)])

    # White noise, random location
    sim.add_noise_sources(
        location=select_random, location_params=dict(n=2), waveform=white_noise
    )

    # Point sources
    sim.add_point_sources(
        location=select_random,
        location_params=dict(n=3),
        waveform=narrowband_oscillation,
        waveform_params=dict(fmin=8, fmax=12),
        snr=[0.5, 1, 5],
        snr_params=dict(fmin=8, fmax=12),
        names=["point1", "point2", "point3"],
    )

    # Patch sources
    sim.add_patch_sources(
        location=[(0, [0, 1, 2]), (0, [3, 4]), (1, [0, 2, 3])],
        location_params=dict(n=3),
        waveform=narrowband_oscillation,
        waveform_params=dict(fmin=8, fmax=12),
        snr=[0.5, 1, 5],
        snr_params=dict(fmin=8, fmax=12),
        extents=None,
        names=["patch1", "patch2", "patch3"],
    )

    # Define several edges to test graph traversal and built-in coupling methods
    sim.set_coupling(("point1", "point2"), method=ppc_constant_phase_shift, phase_lag=0)
    sim.set_coupling(
        coupling={
            ("point2", "patch3"): dict(kappa=0.1, phase_lag=-np.pi / 6),
            ("patch1", "point2"): dict(kappa=1, phase_lag=np.pi / 2),
            ("patch2", "patch3"): dict(kappa=10, phase_lag=2 * np.pi / 3),
        },
        method=ppc_von_mises,
        fmin=8,
        fmax=12,
    )

    # Actual simulation
    sc = sim.simulate(sfreq, duration, fwd=fwd, random_state=seed)

    # SourceConfiguration methods
    stc = sc.to_stc()
    raw = sc.to_raw(fwd, info)
    sc.to_raw(fwd, info, sensor_noise_level=0.25)

    # Check that it is possible to simulate data multiple times
    sc_new = sim.simulate(sfreq, duration, fwd=fwd, random_state=seed)
    stc_new = sc_new.to_stc()
    raw_new = sc_new.to_raw(fwd, info)

    # Check that the result is reproducible
    assert np.allclose(stc.data, stc_new.data)
    assert np.allclose(raw.get_data(), raw_new.get_data())

    # Check that the global adjustment of SNR works
    sim.snr_mode = "global"
    sc = sim.simulate(
        sfreq,
        duration,
        snr_global=5,
        snr_params=dict(fmin=8, fmax=12),
        fwd=fwd,
        random_state=seed,
    )
