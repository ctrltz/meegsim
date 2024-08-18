import numpy as np
from harmoni.extratools import compute_plv
from matplotlib import pyplot as plt
from meegsim.coupling import ppc_von_mises
from meegsim.utils import get_sfreq


lenghts = [.5, 1, 10, 60]
n_kappas = 100
n_runs = 1000
kappas = np.logspace(-5, 3, n_kappas)
fs = 1000
for length in lenghts:
    print(length)
    times = np.arange(0, length, 1 / fs)
    waveform = np.sin(2 * np.pi * 10 * times)
    phase_lag = 0
    plv = np.zeros((n_runs, n_kappas))
    for run in range(n_runs):
        for ikappa, kappa in enumerate(kappas):
            result = ppc_von_mises(waveform, get_sfreq(times), phase_lag, kappa=kappa)
            cplv = compute_plv(waveform, result, m=1, n=1, plv_type='complex')
            plv[run, ikappa] = np.abs(cplv)[0][0]

    plt.plot(np.log10(kappas), plv.T, c='grey')
    plt.plot(np.log10(kappas), np.mean(plv, axis=0), c='k', linewidth=3)
    plt.xlabel('Kappa, logscale')
    plt.ylabel('plv')
    plt.savefig('kappa-to-plv_length' + str(length) + 'sec.png')
    plt.close()
