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
        waveform=np.ones((3, 1000))
        ...
    )

The alternative way is to use built-in or custom functions for setting the 
waveforms of source activity. Since the current focus of the toolbox is on 
simulating connectivity, the two main waveforms are:

* :meth:`narrowband_oscillation` - e.g., for simulating alpha or beta activity

* :meth:`one_over_f_noise` - for adding background noise with the power-law spectra.

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
        waveform_params=dict(fmin=8, fmax=12)
        ...
    )

As mentioned previously for location, the waveforms defined using a function will
differ between different simulations.

.. note::
    Find more details on the built-in template waveforms :doc:`here </api/waveform>`.

Signal-to-noise ratio
---------------------


Source names
------------


Coupling between sources
========================