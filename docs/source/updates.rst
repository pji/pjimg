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
        *   Created sources.NoiseTensor.
        
        *   (Not started.) Create sources.UnitNoissTorch.
        *   (Started.) Create sources.PerlinTorch.
        *   (Not started.) Create relevant unit tests.
        *   (Not started.) Figure out when GPU targeting
            is most valuable.
        *   (Not started.) Convert use case by use case to
            confirm the benefits of the change.
