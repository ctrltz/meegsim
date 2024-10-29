================
General workflow
================

Prerequisites
=============

MEEGsim re-uses the forward and inverse modeling functionality provided
by MNE-Python, so to perform simulations with MEEGsim, you will need:

* A source space (:class:`mne.SourceSpaces`) that contains all possible
  locations for simulated sources. The source space is required for 
  every simulation.

* A forward model (:class:`mne.Forward`) that contains all sources in the
  source space mentioned above. The forward model is required only if
  you wish to obtain sensor-space data or adjust the sensor-space SNR
  of simulated sources.

* An :class:`mne.Info` structure describing the channel layout is required
  for projecting the simulated source activity to sensor space.

Overview
========

MEEGsim provides two main classes that should be used for performing the 
simulations:

.. currentmodule:: meegsim.simulate

* With the :class:`SourceSimulator` class, you can add sources of activity
  to the simulation, configure their signal-to-noise ratio and set up 
  coupling between sources. The class does not contain the simulated data. 
  Instead, it should be used to generate different configurations of sources 
  (see below) as `instances` of the simulation (e.g., if some of the sources 
  are placed in random locations).

.. currentmodule:: meegsim.configuration

* The :class:`SourceConfiguration` class actually contains the simulated data.
  We store the waveforms of all sources separately to ensure that they are 
  always accessible (e.g., in case two sources overlap in space) but also
  provide methods for converting the configuration to commonly used
  ``stc`` and ``raw`` objects for source and sensor spaces, respectively.

Next step
=========

Learn how to describe the simulation in the 
:doc:`next section </user_guide/get_started/simulation>`.
