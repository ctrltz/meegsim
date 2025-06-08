=========================
Simulating the M/EEG data
=========================

Obtaining the data
==================

.. currentmodule:: meegsim.simulate

Up to this point, we defined a bunch of sources and set up several coupling edges.
However, no data was generated yet, and it's time to fix that now.

First, you need to run the :meth:`SourceSimulator.simulate()` method of the ``sim``
object to actually simulate the waveforms of all previously defined sources:

.. code-block:: python

    sfreq = 250     # in Hz
    duration = 30   # in seconds
    sc = sim.simulate(sfreq, duration)

.. currentmodule:: meegsim.configuration

The result of this function call is a :class:`SourceConfiguration` object that
contains all simulated sources and their waveforms (with desired SNR and coupling).

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

.. currentmodule:: meegsim.simulate

By design, the result of :meth:`SourceSimulator.simulate` will differ every time you call the method,
making it very easy to simulate multiple datasets under the same settings.

However, it is always possible to obtain a reproducible result if you provide a
specific value of the ``random_state`` as shown below:

.. code-block:: python

    sc = sim.simulate(sfreq, duration, random_state=123)

The resulting source configuration, including the locations and waveforms of all
sources, will be the same on every call.

Accessing the simulated sources
===============================

The simulated sources can be quickly accessed by their name (`"source"` in the
example below). This might be helpful in case ground-truth waveforms were
generated randomly:

.. code-block:: python

    s = sc["source"]
    gt = s.waveform
