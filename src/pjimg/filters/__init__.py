"""
filters
~~~~~~~

A Python package for distorting or otherwise affecting image data.

.. automodule:: pjimg.filters.filters_
.. automodule:: pjimg.filters.blurs
.. automodule:: pjimg.filters.decorators

"""
import pjimg.filters.blurs as blurs
import pjimg.filters.filters_ as filters_
from pjimg.filters.blurs import box_blur, gaussian_blur, glow, motion_blur
from pjimg.filters.filters_ import *
from pjimg.filters.decorators import *
