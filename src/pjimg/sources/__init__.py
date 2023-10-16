"""
sources
~~~~~~~

Image and video data generation.


Basic Usage: Sources
====================
The image generation source classes (sources) are used to create image
data. Using a source is a two-step process: initialize the source then
generate the data with :meth:`pjimg.sources.Source.fill`.

Usage::

    >>> box = Box((0, 1, 1), (1, 2, 2), 1.0)
    >>> box.fill((1, 4, 4))
    array([[[0., 0., 0., 0.],
            [0., 1., 1., 0.],
            [0., 1., 1., 0.],
            [0., 0., 0., 0.]]])


Source Classes
==============
An image data source is a class with a :meth:`Source.fill` method that
generates image data.

.. autofunction:: pjimg.sources.Source

.. automodule:: pjimg.sources.patterns
.. automodule:: pjimg.sources.noise
.. automodule:: pjimg.sources.unitnoise
.. automodule:: pjimg.sources.perlin
.. automodule:: pjimg.sources.maze
.. automodule:: pjimg.sources.worley

"""
from pjimg.sources.model import Seed, Source
from pjimg.sources.patterns import *
from pjimg.sources.maze import Maze, AnimatedMaze, SolvedMaze
from pjimg.sources.noise import Embers, Noise
from pjimg.sources.perlin import OctavePerlin, Perlin
from pjimg.sources.unitnoise import *
from pjimg.sources.worley import OctaveWorley, Worley
