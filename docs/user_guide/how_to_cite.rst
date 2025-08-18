How to cite
===========

If you used MEEGsim in your project, please consider citing the software using the
DOI provided by Zenodo: https://doi.org/10.5281/zenodo.15106042

.. code-block::

    @software{MEEGsim,
        author = {Kapralov, Nikolai and Studenova, Alina and Jamshidi Idaji, Mina},
        license = {BSD-3-Clause},
        title = {{MEEGsim}},
        doi = {https://doi.org/10.5281/zenodo.15106042},
        version = {0.0.2},
        year = {2025}
    }

In addition, please `cite MNE-Python <https://mne.tools/stable/documentation/cite.html>`_
since MEEGsim re-uses lots of important functionality
provided by this toolbox.

Finally, check the section below to find reference papers for methods behind MEEGsim.

Methods
-------

Below we also provide reference papers for some of the approaches and functions provided by MEEGsim.
By citing these references, you can guide the reader to the description of underlying methods:

+------------------------------------------------------+------------------------------------------+
| **Approach / Function**                              | **Reference**                            |
+------------------------------------------------------+------------------------------------------+
| Global adjustment of the SNR                         | :footcite:`HaufeEwald2019`               |
+------------------------------------------------------+------------------------------------------+
| Local adjustment of the SNR                          | :footcite:`Nikulin2011`                  |
+------------------------------------------------------+------------------------------------------+
| :py:func:`meegsim.coupling.ppc_constant_phase_shift` | :footcite:`JamshidiIdaji2022_PhDThesis`  |
+------------------------------------------------------+------------------------------------------+
| :py:func:`meegsim.coupling.ppc_von_mises`            | :footcite:`JamshidiIdaji2022_PhDThesis`  |
+------------------------------------------------------+------------------------------------------+


References
----------

.. footbibliography::
