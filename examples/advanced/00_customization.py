"""
Using functions from other packages
-----------------------------------

This example show how to adapt function from other packages to
use them with MEEGsim.
"""

from neurodsp.sim import sim_bursty_oscillation
from neurodsp.sim.multi import sim_multiple

from meegsim.simulate import SourceSimulator
from meegsim.utils import normalize_variance


def bursty_osc(n_series, times, **kwargs):
    # Convert MEEGsim input to NeuroDSP input
    n_seconds = times.max()
    fs = 1.0 / (times[1] - times[0])

    params = dict(n_seconds=n_seconds, fs=fs)
    params.update(kwargs)

    sims = sim_multiple(sim_bursty_oscillation, **params, n_sims=n_series)

    return normalize_variance(sims.signals)


# src should be loaded before
src = None
sim = SourceSimulator(src)

sim.add_point_sources(
    location=[(0, 123), (1, 456)],
    waveform=bursty_osc,
    waveform_params=dict(  # NeuroDSP parameters
        freq=20,
        burst_def="durations",
        burst_params={"n_cycles_burst": 3, "n_cycles_off": 3},
    ),
    # snr / std / names
)
