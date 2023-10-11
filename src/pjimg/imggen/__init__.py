"""
imggen
~~~~~~

Image and video data generation.

Sources
=======
An image data source is a class with a :meth:`Source.fill` method that
generates image data.

.. autofunction:: pjimg.imggen.Source

.. automodule:: pjimg.imggen.patterns
.. automodule:: pjimg.imggen.noise
"""
from pjimg.imggen.model import Seed, Source
from pjimg.imggen.patterns import *
from pjimg.imggen.noise import Embers, Noise
