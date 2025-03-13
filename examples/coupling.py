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
            coh_sim[i_lag, i_coh, i_sim] = compute_plv(
                x, y, n=1, m=1, plv_type="complex", coh=True
            )[0, 0]

target_values = np.tile(target_coherence[:, np.newaxis], (1, n_simulations))
fig, axes = plt.subplots(ncols=n_lags)
for ax, lag in zip(axes, phase_lags):
    ax.scatter(target_values.flatten(), np.abs(coh_sim[i_lag, :, :].flatten()))
    ax.plot([0, 1], [0, 1], c="gray", ls="--")
plt.show(block=True)
