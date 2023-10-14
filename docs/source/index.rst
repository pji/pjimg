.. pjimg documentation master file, created by
   sphinx-quickstart on Tue Oct 10 16:39:03 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pjimg
~~~~~
Procedural image generation.


.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   self
   /imgblend.rst
   /imgease.rst
   /imgfilt.rst
   /imggen.rst
   /imgio.rst
   /util.rst
   /requirements.rst


What can I do with this?
========================
Great question! You can use :mod:`pjimg` to generate images and video. For
examples, see the scripts in the `examples` folder in the directory.


Why did you write this?
=======================
The commercial software I used to generate "clouds" (Perlin noise)
moved to a subscription model. I then switched to a different commercial
software package, but it's Perlin noise generation was pretty weak. So,
I decided to write something to make it myself. And, the scope, as it
tends to do, grew.


How do I run this?
==================
You can clone the repository to your local system, then install it with
`pip`::

    pip3 install /path/to/repo/pjimg

Replace '/path/to/repo` with the path to the repository on your local
system. You should then be able to import it into the python scripts
you write.

For examples for what to do from there, look at the `examples` folder
in the repository.


How do I run the tests?
=======================
The `precommit.py` script in the root of the repository will run the
unit tests and a few other tests beside. Otherwise, the unit tests
are written with the `pytest` module, so you can run the tests with::

    python -m pytest


How do I contribute?
====================
At this time, this is code is really just me exploring and learning.
I've made it available in case it helps anyone else, but I'm not really
intending to turn this into anything other than a personal project.

That said, if other people do find it useful and start using it, I'll
reconsider. If you do use it and see something you want changed or
added, go ahead and open an issue. If anyone ever does that, I'll
figure out how to handle it.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
