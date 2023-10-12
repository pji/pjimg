"""
imggen
~~~~~~

Image and video data generation.

sources
=======
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
