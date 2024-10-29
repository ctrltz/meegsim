=========================
Describing the simulation
=========================

.. currentmodule:: meegsim.simulate

Initialization
==============

The starting point of any simulation is to initialize the :class:`SourceSimulator`
and provide an :class:`mne.SourceSpaces` object that describes the source space:

.. code-block:: python
    
    from meegsim.simulate import SourceSimulator
    
    # src should be loaded or created beforehand
    sim = SourceSimulator(src)

The ``sim`` object defined above can be used to add sources to the simulation and 
set ground truth connectivity patterns as shown below.

Types of sources
================

Currently, there are several types of sources that could be added to the simulation
using different methods:

* Point-like sources (:meth:`SourceSimulator.add_point_sources`): the time course 
  of activity is assigned to one vertex.

* Cortical patches (:meth:`SourceSimulator.add_patch_sources`): the time course of 
  activity is assigned to a set of vertices.

.. note::
    Currently, all vertices belonging to the patch have identical waveforms.

* Noise sources (:meth:`SourceSimulator.add_noise_sources`): these sources are 
  automatically considered as `noise` when adjusting the SNR, while point-like 
  or patch sources are treated as `signal`.

.. note::
    Currently, only point-like noise sources are supported.

Adding sources to the simulation
================================

To define the source(s), you need to call the corresponding method of the ``sim``
object:

.. code-block:: python

    sim.add_point_sources(...)
    sim.add_patch_sources(...)
    sim.add_noise_sources(...)

The parameters in brackets can be used to configure the location of the source, 
the waveform of source activity, the SNR and, for patch sources only, the extent
of the source. Below we discuss the parameters relevant for all sources in more 
detail. For more information about patch sources, please visit 
:doc:`this page </user_guide/advanced/patches>`.

Source location
---------------

.. currentmodule:: meegsim.location

The location of the sources can be specified using the ``location`` argument.
You can either directly provide the indices of vertices where the sources should
be placed or pass a function that returns such indices.

In the first case, you need to provide a list of 2-element tuples, where the 
first element specifies the index of the source space, while the second contains
the index of the vertex (`vertno`). If we consider the typical surface source 
space with two hemispheres, the call below will add two sources to the left 
hemisphere (123 and 456) and one source to the right hemisphere (789).

.. code-block:: python

    sim.add_point_sources(
        location=[(0, 123), (0, 456), (1, 789)],
        ...
    )

.. warning::
    Note that the format is slightly different from the list of lists commonly used
    in MNE-Python. The MNE format assumes sorting of vertex indices belonging to 
    the same hemisphere, which might interfere with the idea of source names 
    needed for convenient setup of coupling (see `Source names`_).

In the second case, you need to provide a function that accepts ``src`` as the first
argument and returns the indices of selected vertices in the same format as 
described above. If the function relies on additional arguments, these can be 
provided in the ``location_params`` argument. In the example below, 
we use a built-in function :meth:`select_random` to place 10 sources in random
locations:

.. code-block:: python

    from meegsim.location import select_random

    sim.add_point_sources(
        location=select_random,
        location_params=dict(n=10),
        ...
    )

As one of the benefits of specifying location as a function, the simulated source
configurations may differ between subsequent calls of the same ``sim`` object, 
simplifying the simulation of multiple datasets with the same underlying idea.

.. note::
    Find more details on the built-in location functions :doc:`here </api/location>`.

Activity waveforms
------------------

.. currentmodule:: meegsim.waveform

Similar to location, activity waveforms can also be defined manually or through 
a function. In the first case, the toolbox expects an array with one time series 
per added source, and the number of sources should match the number of defined 
locations. We can now add some activity to the point sources described above 
(here we use constant time series for simplicity):

.. code-block:: python

    import numpy as np
    
    sim.add_point_sources(
        location=[(0, 123), (0, 456), (1, 789)],
        waveform=np.ones((3, 1000)),
        ...
    )

The alternative way is to use built-in or custom functions for setting the 
waveforms of source activity. Since the current focus of the toolbox is on 
simulating connectivity, the two main waveforms are:

* :meth:`narrowband_oscillation` - e.g., for simulating alpha or beta activity

* :meth:`one_over_f_noise` - for adding background noise with the power-law spectra.
  This waveform is used for all noise sources by default.

Additional parameters for the waveform function can be provided using the
``waveform_params`` argument. Let's now add an alpha (8-12 Hz) oscillation 
to the point sources from the second example:

.. code-block:: python

    from meegsim.location import select_random
    from meegsim.waveform import narrowband_oscillation

    sim.add_point_sources(
        location=select_random,
        location_params=dict(n=10),
        waveform=narrowband_oscillation,
        waveform_params=dict(fmin=8, fmax=12),
        ...
    )

As mentioned previously for location, the waveforms defined using a function will
differ between simulated configurations by default.

.. note::
    Find more details on the built-in template waveforms :doc:`here </api/waveform>`.

Signal-to-noise ratio (SNR)
---------------------------

By default, the waveforms provided by the user are saved as is, while the built-in 
waveforms are normalized to have the same variance. In practice, it is often useful
to set up a specific SNR for each source, for example, to test how different analysis
methods perform depending on the SNR of target sources. 

We use the approach from :cite:p:`Nikulin2011` for adjusting the sensor-space SNR 
of sources. Namely, we calculate the mean variance of each point or patch source 
across all sensors and adjust it relative to the mean variance of all noise sources
(mean over sensors but summed over noise sources). The calculation of variance is 
performed after filtering both time series (signal and noise) in the frequency 
band of interest.

By default, no adjustment of SNR is performed. To enable it, you need to specify
the value of SNR using the ``snr`` argument and provide the limits of the frequency 
band in ``snr_params`` when adding the point or patch sources:

.. code-block:: python

    import numpy as np
    
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
    
    sim.add_point_sources(
        location=[(0, 123), (0, 456), (1, 789)],
        waveform=np.ones((3, 1000)),
        snr=[1, 2.5, 5],
        snr_params=dict(fmin=8, fmax=12),
        ...
    )

Configuring coupling between sources
====================================

With the toolbox, we aim to provide a convenient interface for the generation of 
source waveforms with desired coupling. To set the coupling between sources, you 
only need to specify the names of sources that should be coupled and the coupling 
parameters. The waveforms will be then generated automatically to obtain the 
desired coupling.

Source names
------------

An auto-generated name is assigned to all sources that are added to the simulation.
However, it might be more convenient to set up a custom and more meaningful name.
For this, you can use the ``names`` argument when adding sources, e.g.:

.. code-block:: python

    import numpy as np
    
    names = sim.add_point_sources(
        location=[(0, 123), (0, 456), (1, 789)],
        waveform=np.ones((3, 1000)),
        snr=[1, 2.5, 5],
        snr_params=dict(fmin=8, fmax=12),
        names=['source', 'sink', 'other']
    )

The provided or the auto-generated names are always returned if the 
sources were added successfully.

Specifying the coupling parameters
----------------------------------

.. currentmodule:: meegsim.simulate

To specify which sources should be coupled, you can use the 
:meth:`SourceSimulator.set_coupling` method. Provide the names
of sources to be coupled as a tuple ``(u, v)`` along with the 
coupling parameters as shown below:

.. code-block:: python

    from meegsim.coupling import constant_phase_shift

    sim.set_coupling(('source', 'sink'),
        method=constant_phase_shift, phase_lag=np.pi/3
    )

.. currentmodule:: meegsim.coupling

.. note::
    Find more details on the parameters of built-in coupling methods
    :doc:`here </api/coupling>`.

In the example above, we used the :meth:`constant_phase_shift` coupling method.
As the name suggests, it generates time series with a constant phase shift 
relative to each other. To have more control over the strength of coupling,
you can use the :meth:`ppc_von_mises` method that set probabilistic phase shifts
based on the von Mises distribution. For more details about both approaches,
see chapter 3 of :cite:`JamshidiIdaji2022_PhDThesis`.

Finally, it is possible to defined multiple coupling edges with one call.
For this, you can provide a dictionary with edges as keys and parameters as
values:

.. code-block:: python

    from meegsim.coupling import ppc_von_mises

    sim.set_coupling(coupling={
            ('source', 'sink'): dict(kappa=1, phase_lag=0),
            ('source', 'other'): dict(kappa=0.1, phase_lag=np.pi/3),
        },
        method=ppc_von_mises, fmin=8, fmax=12
    )

Parameters specified outside the dictionary (``method``, ``fmin``, and ``fmax``
in the example above) apply to all coupling edges, while
parameters in the dictionary (``kappa`` and ``phase_lag``) will apply only 
to the corresponding edges and have higher priority when specified.

.. note::
    We make sure that the coupling is set up properly regardless of the order,
    in which the coupling edges were defined. However, the cycles in the 
    coupling graph are currently not supported.

Next step
=========

Learn how to obtain the M/EEG data for the defined simulation in the
:doc:`next section </user_guide/get_started/configuration>`.


References
==========

.. bibliography::