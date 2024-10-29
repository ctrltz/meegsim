=========================
Simulating the M/EEG data
=========================

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
===============

By design, the subsequent ``simulate()`` call
