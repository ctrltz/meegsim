"""
Phase-phase coupling using shifted copy with noise
==================================================
"""

import numpy as np
import matplotlib.pyplot as plt

from harmoni.extratools import compute_plv
from meegsim.coupling import ppc_shifted_copy_with_noise
from meegsim.waveform import narrowband_oscillation


sfreq = 250
duration = 120
fmin = 8
fmax = 12
seed = 1234

phase_lags = [0, np.pi / 4, np.pi / 2]
target_coherence = np.linspace(0, 1, num=11)
n_lags = len(phase_lags)
n_cohs = target_coherence.size
n_simulations = 25

times = np.arange(sfreq * duration) / sfreq
x = narrowband_oscillation(1, times, fmin=fmin, fmax=fmax, random_state=seed)

seeds = np.random.SeedSequence(seed).generate_state(n_simulations)
coh_sim = np.zeros((n_lags, n_cohs, n_simulations), dtype=np.complex128)
for i_lag, lag in enumerate(phase_lags):
    for i_coh, target_coh in enumerate(target_coherence):
        for i_sim, seed in enumerate(seeds):
            y = ppc_shifted_copy_with_noise(
                x, sfreq, target_coh, lag, fmin, fmax, random_state=seed
            )
            # x is leading y
            coh_sim[i_lag, i_coh, i_sim] = compute_plv(
                y, x, n=1, m=1, plv_type="complex", coh=True
            )[0, 0]

target_values = np.tile(target_coherence[:, np.newaxis], (1, n_simulations))
fig, axes = plt.subplots(nrows=2, ncols=n_lags, figsize=(9, 6))
for i_lag, lag in enumerate(phase_lags):
    lag_in_degrees = np.rad2deg(lag)
    obtained_coh = np.abs(coh_sim[i_lag, :, :])
    obtained_lag = np.rad2deg(np.angle(coh_sim[i_lag, :, :]))
    mean_coh = obtained_coh.mean(axis=1)
    mean_lag = obtained_lag.mean(axis=1)

    ax_coh = axes[0, i_lag]
    ax_coh.scatter(target_values.flatten(), obtained_coh.flatten(), c="gray", alpha=0.1)
    ax_coh.plot(target_coherence, mean_coh, c="black", lw=2)
    ax_coh.set_xlim([0, 1])
    ax_coh.set_ylim([0, 1])
    ax_coh.set_xlabel("Target coherence")
    ax_coh.set_ylabel("Obtained coherence")
    ax_coh.set_title(f"phase lag = {lag_in_degrees:.0f} degrees")

    ax_lag = axes[1, i_lag]
    ax_lag.scatter(target_values.flatten(), obtained_lag.flatten(), c="gray", alpha=0.1)
    ax_lag.plot(target_coherence, mean_lag, c="black", lw=2)
    ax_lag.set_xlim([0, 1])
    ax_lag.set_ylim([-180, 180])
    ax_lag.set_xlabel("Target coherence")
    ax_lag.set_ylabel("Obtained phase lag (degrees)")
fig.tight_layout()
plt.show(block=True)
