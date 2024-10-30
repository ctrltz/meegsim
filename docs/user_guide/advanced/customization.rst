=============
Customization
=============

If the built-in functions for location, waveform or coupling do not satisfy the 
needs of your project, you are always welcome use a custom function instead. 
In this section, we provide a short description of requirements for such functions:

* which arguments should be accepted (but not necessarily used)

* the expected format of the returned value

In addition, we provide minimal examples of custom functions for each case.

.. note::
    When adding sources, we always try to execute the provided functions with 0 
    as ``random_state`` to making the debugging a bit easier.

Location
========

The location function should accept the source space as the first argument and 
have a ``random_state`` keyword argument.

The return value is expected to be a list of 2-element tuples (index of the
source space, index of the selected vertex).

For example, the function below will pick the vertex with the smallest index 
(`vertno`) from the provided ``src``:

.. code-block:: python

    def pick_first(src, src_idx, random_state=None):
        picked = src[src_idx]['vertno'][0]
        return [(src_idx, picked)]

Since the result is deterministic, the ``random_state`` argument will have no
effect, and the same vertex will be used in all simulated configurations.

The function could be used in the following way:

.. code-block:: python

    sim.add_point_sources(
        location=pick_first,
        location_params=dict(src_idx=0),
        ...
    )

Waveform
========

The waveform function should accept:

* the number of time series to generate

* the array of time points (in seconds) that the generated samples should 
  correspond to

* keyword argument ``random_state``

The result is expected to be an array with shape ``(n_series, n_times)``.

The function below returns white noise, and it produces different results every
time unless ``random_state`` is fixed:

.. code-block:: python

    import numpy as np

    def my_white_noise(n_series, times, random_state=None):
        rng = np.random.default_rng(seed=random_state)
        return rng.random((n_series, len(times))

Let's now add this function to the location example (note that ``waveform_params``
are not required in this case):

.. code-block:: python

    sim.add_point_sources(
        location=pick_first,
        location_params=dict(src_idx=0),
        waveform=my_white_noise
    )

Coupling
========

The coupling function should accept:

* an input waveform as an array with shape ``(n_times,)``

* its sampling frequency

* keyword argument ``random_state``

As a result, this function should return another waveform of the same length.

Below is an example function that returns a scaled copy of the input waveform:

.. code-block:: python

    def scaled_copy(waveform, sfreq, scaling_factor=1, random_state=None):
        return scaling_factor * waveform

The function could be used like this:

.. code-block:: python

    sim.set_coupling(
        ('s1', 's2'), 
        method=scaled_copy, scaling_factor=2
    )

Extending the toolbox
=====================

If you think that your custom function could be helpful for others, feel free to 
open an issue in the `GitHub repository <https://github.com/ctrltz/meegsim>`_.
