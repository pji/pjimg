"""
filters
~~~~~~~

A Python package for distorting or otherwise affecting image data.

.. automodule:: pjimg.filters.filts
.. automodule:: pjimg.filters.decorators

"""
import pjimg.filters.filters_ as filters_
from pjimg.filters.filters_ import *
from pjimg.filters.decorators import *
from pjimg.util import get_prefixed_functions as _get_prefixed_functions


# Mapping of registered eases.
filters = _get_prefixed_functions('filter_', filters_)
