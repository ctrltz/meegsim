=============
Patch sources
=============

Since the getting started tutorial was focused on point sources, here we provide
more information that is specific for patch sources. In particular, there are two
ways to construct such sources, which we describe below.

Specifying all vertices belonging to the patch (default)
========================================================

By default, you are expected to provide indices of all vertices that belong to the
patch in the second element of ``location`` tuples.

For example, below we define a patch that contains three vertices:

.. code-block:: python

    sim.add_patch_sources(
        location=[(0, [123, 234, 345])],
        ...
    )

And here we define two patches (with 3 and 4 vertices, respectively) in
the same call:

.. code-block:: python

    sim.add_patch_sources(
        location=[
            (0, [123, 234, 345]),
            (1, [0, 1, 2, 3])
        ],
        ...
    )


Growing patches from the center
===============================

Alternatively, you can provide a vertex that should be used as the center of the
patch and specify patch radius (in millimeters) in the ``extents`` argument.
In this case, we use the :func:`mne.grow_labels` function to construct the patch.

Example 1 - single patch:

.. code-block:: python

    sim.add_patch_sources(
        location=[(0, 123)],
        extent=15             # in mm
    )

Example 2 - several patches of different size:

.. code-block:: python

    sim.add_patch_sources(
        location=[(0, 123), (1, 456)],
        extent=[15, 30]       # in mm
    )
