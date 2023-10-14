"""
imggen
~~~~~~

Image and video data generation.


Basic Usage: Sources
====================
The image generation source classes (sources) are used to create image
data. Using a source is a two-step process: initialize the source then
generate the data with :meth:`pjimg.imggen.Source.fill`.

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

.. autofunction:: pjimg.imggen.Source

.. automodule:: pjimg.imggen.patterns
.. automodule:: pjimg.imggen.noise
.. automodule:: pjimg.imggen.unitnoise
.. automodule:: pjimg.imggen.perlin
.. automodule:: pjimg.imggen.maze
.. automodule:: pjimg.imggen.worley

"""
from pjimg.imggen.model import Seed, Source
from pjimg.imggen.patterns import *
from pjimg.imggen.maze import Maze, AnimatedMaze, SolvedMaze
from pjimg.imggen.noise import Embers, Noise
from pjimg.imggen.perlin import OctavePerlin, Perlin
from pjimg.imggen.unitnoise import *
from pjimg.imggen.worley import OctaveWorley, Worley
