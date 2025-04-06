=====================================
Standard deviation of source activity
=====================================

By default, all generated waveforms (but not the ones explicitly provided by the user)
are normalized to have a standard deviation of 1 nAm. If necessary, this value can be
modified using the ``base_std`` argument when initializing the ``SourceSimulator``:

.. code-block:: python

    from meegsim.simulate import SourceSimulator

    sim = SourceSimulator(src, base_std=1e-8)    # 10 nAm

Normalization also means that the standard deviation of activity will be the same for
all added sources by default. However, in real data this does not always hold.
For example, alpha power is usually higher in parieto-occipital areas. Therefore, we
also provide an option to configure the standard deviation of activity for each
source when adding:

.. code-block:: python

    sim.add_point_sources(
        location=...,
        waveform=...,
        std=5,                # 5 times stronger than by default
    )

.. note::
    :doc:`Local adjustment of SNR </user_guide/advanced/local_snr>` will have priority
    over the provided value of standard deviation.

The standard deviation can also be set for all candidate source locations
using an instance of :py:class:`mne.SourceEstimate`. If the source locations are
generated randomly, rhe standard deviation will be adjusted according to the
generated location of the source.
