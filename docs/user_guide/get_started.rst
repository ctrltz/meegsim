============================
Getting started with MEEGsim
============================

.. currentmodule:: meegsim.simulate

Welcome to MEEGsim! In this tutorial, you will learn how to use our toolbox to
simulate an M/EEG dataset. In particular, we will cover the following aspects:

* prerequisites for the simulation
* different types of sources and how to add them to the simulation
* how to set up location and waveform of any source
* how to adjust the signal-to-noise ratio (SNR) of added sources
* how to set up phase coupling between waveforms of source activity

Initialization
==============

The starting point of any simulation is to initialize the :class:`SourceSimulator`
and provide an :class:`mne.SourceSpaces` object that describes the source space:

.. code-block:: python
    
    from meegsim.simulate import SourceSimulator
    
    sim = SourceSimulator(src)

The ``sim`` object defined above can be used to add sources to the simulation and 
set ground truth connectivity patterns as shown below.

Types of sources
================

Currently, there are several types of sources that could be added to the simulation:

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
of the source. Below we discuss these parameters in more detail.

Source location
---------------

.. currentmodule:: meegsim.location

The location of the sources can be specified using the ``location`` argument.
You can either directly provide the indices of vertices for the sources or 
pass a function that returns such indices.

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
    the same hemisphere, which might interfere with source names described below.

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
differ between different simulations.

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
across all sensors and adjust it relative to the mean variance of all noise sources.
The calculation of variance is performed after filtering both time series (signal
and noise) in the frequency band of interest.

By default, no adjustment of SNR is performed. To enable it, you need to specify
the value of SNR using the ``snr`` argument and provide the limits of the frequency 
band in ``snr_params`` as shown below:

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
parameters. The waveforms will be then generated automatically according to the 
provided parameters.

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

The provided or the auto-generated names are always returned as the result if the 
sources were added successfully.

Specifying the coupling parameters
----------------------------------

To 

Multiple edges can be defined with one call

Obtaining the data
==================

.. currentmodule:: meegsim.simulate

Up to this point, we defined a bunch of sources and set up several coupling links.
However, no data was generated yet, and it's time to fix that now.

First, you need to run the :meth:`SourceSimulator.simulate()` method of the ``sim`` 
object to actually simulate the waveforms of all previously defined sources:

.. code-block:: python
    
    sfreq = 250     # in Hz
    duration = 30   # in seconds
    sc = sim.simulate(sfreq, duration)

.. currentmodule:: meegsim.configuration

The result of this function call is a :class:`SourceConfiguration` object that 
contains all simulated sources. 

Now you can use the :meth:`SourceConfiguration.to_stc()` and 
:meth:`SourceConfiguration.to_raw()` to obtain source time courses and 
sensor-space data, respectively. The projection to sensor space requires a 
forward model (:class:`mne.Forward`) and an :class:`mne.Info` object describing
the sensor layout:

.. code-block:: python

    stc = sc.to_stc()

    raw = sc.to_raw(fwd, info)

Reproducibility
---------------

By design, the subsequent ``simulate()`` call

Full example
============

.. code-block:: python

    import numpy as np

    from meegsim.coupling import ppc_von_mises
    from meegsim.location import select_random
    from meegsim.simulate import SourceSimulator
    from meegsim.waveform import narrowband_oscillation

    
    # You need to load the prerequisites: fwd, src, and info

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


.. include:: ../bibliography.rst