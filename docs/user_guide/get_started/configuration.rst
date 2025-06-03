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

Global adjustment of the SNR
----------------------------

If necessary, the global SNR can be adjusted when simulating the data. To achieve this,
you need to specify the desired value of SNR (``snr_global``) and the limits of
the frequency band that should be considered during the adjustment (``fmin`` and
``fmax`` keys in the ``snr_params`` dictionary). As a result, the mean sensor-space
variance of all point and patch sources will be adjusted relative to the mean variance
of all noise sources to achieve the desired SNR:

.. code-block:: python

    sfreq = 250     # in Hz
    duration = 30   # in seconds
    sc = sim.simulate(
        sfreq,
        duration,
        fwd=fwd,    # Forward model is required for the adjustment of SNR
        snr_global=3.0,
        snr_params=dict(fmin=8, fmax=12)
    )

Adding sensor noise
-------------------

When projecting the simulated source activity to sensor space, it is also possible
to add a desired amount of sensor noise (currently modeled as white noise):

.. code-block:: python

    raw = sc.to_raw(fwd, info, sensor_noise_level=0.01)

The ``sensor_noise_level`` controls the ratio of noise to total power in sensor space.
In the example above, 1% of total sensor-space power will originate from noise.

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
