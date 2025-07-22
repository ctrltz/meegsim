"""
Phase-phase coupling based on von Mises distribution
====================================================

This example illustrates how the choice of kappa affects the obtained phase-phase
coupling if the method based on von Mises distribution is used.
"""

import numpy as np
from harmoni.extratools import compute_plv
from matplotlib import pyplot as plt

from meegsim.coupling import ppc_von_mises
from meegsim.waveform import narrowband_oscillation

lengths = [10, 30, 60]
n_kappas = 11
n_runs = 25
kappas = np.logspace(-5, 3, n_kappas)
sfreq = 250
fmin = 8
fmax = 12

fig, axes = plt.subplots(ncols=len(lengths), figsize=(9, 4), layout="constrained")
for ax, length in zip(axes, lengths):
    times = np.arange(0, length, 1 / sfreq)
    waveform = narrowband_oscillation(1, times, fmin=fmin, fmax=fmax)

    phase_lag = np.pi / 4
    plv = np.zeros((n_runs, n_kappas))
    for i_run in range(n_runs):
        for i_kappa, kappa in enumerate(kappas):
            result = ppc_von_mises(
                waveform, sfreq, phase_lag, kappa=kappa, fmin=fmin, fmax=fmax
            )
            cplv = compute_plv(waveform, result, m=1, n=1, plv_type="complex")
            plv[i_run, i_kappa] = np.abs(cplv)[0][0]

    ax.semilogx(kappas, plv.T, c="grey")
    ax.semilogx(kappas, np.mean(plv, axis=0), c="k", linewidth=1.5)
    ax.set_xlabel("Kappa")
    ax.set_ylabel("PLV")
    ax.set_title(f"duration = {length} s")
