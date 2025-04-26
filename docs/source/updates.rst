.. _updates:

Updates
~~~~~~~
The following are notes on the changes in each version of :mod:`pjimg`.


Updates in v0.1.0
=================
The following were the updates in :mod:`pjimg` v0.1.0:

*   Created this update page.
*   Replaced :mod:`pipenv` with :mod:`poetry`.
*   Replaced `precommit.py` with `make` automation.
*   Set up multiversion testing with :mod:`tox`.
*   Refactored pjimg.sources.Perlin for a performance gain.
*   (Started.) Use :mod:`torch.Tensor` objects instead of 
    :mod:`numpy.ndarray` objects when GPU processing would
    be most valuable.

        *   Created time_trail.py example script for comparing
            performance of different source objects.
        *   Added tensor-based generation to all sources.
        
            *   Added to Perlins.
            *   Added to Mazes.
            *   Added to Noises.
            *   Added to UnitNoises.

            *   Added to Worleys.
            *   Added to Tiles.
            *   Added to Patterns.
        
        *   Allow reading from and writing to tensors.

        *   (Not started.) Allow direct filtering of tensors.
        *   (Not started.) Allow direct easing of tensors.
        *   (Not started.) Allow direct blending of tensors.
        *   (Started.) Create relevant unit tests.
