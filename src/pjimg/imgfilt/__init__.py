"""
imgfilt
~~~~~~~

A Python package for distorting or otherwise affecting image data.

.. automodule:: pjimg.imgfilt.filts
.. automodule:: pjimg.imgfilt.decorators

"""
import pjimg.imgfilt.filts as filts
from pjimg.imgfilt.filts import *
from pjimg.imgfilt.decorators import *
from pjimg.util import get_prefixed_functions as _get_prefixed_functions


# Mapping of registered eases.
filters = _get_prefixed_functions('filter_', filts)
