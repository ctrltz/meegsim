================
General workflow
================

.. currentmodule:: meegsim.simulate

Initialization
==============

The starting point of any simulation is to initialize the :class:`SourceSimulator`
and provide an :class:`mne.SourceSpaces` object that describes the source space:

.. code-block:: python
    
    from meegsim.simulate import SourceSimulator
    
    sim = SourceSimulator(src)

The ``sim`` object defined above can be used to add sources to the simulation and 
set ground truth connectivity patterns as shown below.
