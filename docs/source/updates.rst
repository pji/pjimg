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
        *   (Started.) Added tensor-based generation to all sources.
        
            *   (Started.) Added to Noises.
            *   (Started.) Added to UnitNoises.
            *   Added to Perlins.
            *   (Not started.) Added to Worleys.
            *   Added to Mazes.
            *   (Not started.) Added to Tiles.
            *   (Not started.) Added to Patterns.
        
        *   (Started.) Create relevant unit tests.
