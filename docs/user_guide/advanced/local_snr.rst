=======================
Local adjustment of SNR
=======================

To enable the *local* adjustment of SNR, you need to enable the corresponding ``snr_mode``
when initializing the ``SourceSimulator`` class:

.. code-block:: python

    from meegsim.simulate import SourceSimulator

    # src should be loaded or created beforehand
    sim = SourceSimulator(src, snr_mode="local")

For the *local* adjustment of SNR, we calculate the mean variance of each point or patch source
across all sensors and adjust it relative to the mean variance of all noise sources
(mean over sensors but summed over noise sources). The calculation of variance is
performed after filtering both time series (signal and noise) in the frequency
band of interest.

.. note::
    For the adjustment of SNR, you always need to add noise sources to the
    simulation.

By default, no adjustment of *local* SNR is performed. To enable it, you need to specify
the value of SNR using the ``snr`` argument and provide the limits of the frequency
band in ``snr_params`` when adding the point or patch sources:

.. code-block:: python

    import numpy as np

    # add some noise sources
    sim.add_noise_sources(
        location=select_random,
        location_params=dict(n=10)
    )

    # now add point sources with adjustment of SNR
    sim.add_point_sources(
        location=[(0, 123), (0, 456), (1, 789)],
        waveform=np.ones((3, 1000)),
        snr=5,
        snr_params=dict(fmin=8, fmax=12),
        ...
    )

It is also possible to specify SNR for each of the sources separately by providing
one value for each source:

.. code-block:: python

    import numpy as np

    # add some noise sources first
    sim.add_noise_sources(
        location=select_random,
        location_params=dict(n=10)
    )

    # now add point sources with adjustment of SNR
    sim.add_point_sources(
        location=[(0, 123), (0, 456), (1, 789)],
        waveform=np.ones((3, 1000)),
        snr=[1, 2.5, 5],
        snr_params=dict(fmin=8, fmax=12),
        ...
    )
