A minimal example
=================

Below you can find an example script that contains all the ideas showcased
in the previous sections of the overview. It may serve as a good starting point
for your own simulation.

.. code-block:: python

    import numpy as np

    from meegsim.coupling import ppc_von_mises
    from meegsim.location import select_random
    from meegsim.simulate import SourceSimulator
    from meegsim.waveform import narrowband_oscillation

    
    # Here you need to load the prerequisites: fwd, src, and info

    # Simulation parameters
    sfreq = 250
    duration = 120

    # Initialize
    sim = SourceSimulator(src)

    # Add 500 noise sources with random locations
    sim.add_noise_sources(
        location=select_random,
        location_params=dict(n=500)
    )

    # Add two point sources with fixed locations 
    # (vertex indices are chosen arbitrarily)
    sim.add_point_sources(
        location=[(0, 123), (1, 456)],
        waveform=narrowband_oscillation,
        waveform_params=dict(fmin=8, fmax=12),
        snr=[2, 5],
        snr_params=dict(fmin=8, fmax=12),
        names=['s1', 's2']
    )

    # Set the coupling between point sources
    sim.set_coupling(
        ('s1', 's2'),
        method=ppc_von_mises,
        kappa=1, phase_lag=np.pi/2,
        fmin=8, fmax=12
    )

    # Obtain the data
    sc = sim.simulate(sfreq, duration, fwd, random_state=0)

    stc = sc.to_stc()
    raw = sc.to_raw(fwd, info)
