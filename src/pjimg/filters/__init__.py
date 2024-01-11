"""
filters
~~~~~~~

A Python package for distorting or otherwise affecting image data.

Filters
=======
The filter operation functions (filters) are used to make changes to
values in image data where the resulting value of each pixel can be
influenced by the values of other pixels in the data.

Usage::

    >>> import numpy as np
    >>> a = np.array([[[0., .25, .5, .75, 1.], [0., .25, .5, .75, 1.]]])
    >>> box_blur(a, size=2)
    array([[[0.125, 0.125, 0.375, 0.625, 0.875],
            [0.125, 0.125, 0.375, 0.625, 0.875]]])

The parameters of a filter depends on what the filter is doing. However,
the following is true for all of them:

*   They take a :class:`numpy.ndarray` of image data as the first parameter.
*   They return a :class:`numpy.ndarray` of image data.

The following filters are available in :mod:`pjimg.filters`.


.. automodule:: pjimg.filters.affine
.. automodule:: pjimg.filters.blurs
.. automodule:: pjimg.filters.distort
.. automodule:: pjimg.filters.value


Registration
============
All filter functions are registered in the :class:`dict`
`pjimg.filters.filters` for convenience, but they can also
be called directly.


.. automodule:: pjimg.filters.decorators

"""
import pjimg.filters.affine as affine
import pjimg.filters.blurs as blurs
import pjimg.filters.distort as distort
import pjimg.filters.value as value
from pjimg.filters.affine import *
from pjimg.filters.blurs import *
from pjimg.filters.distort import *
from pjimg.filters.decorators import *
from pjimg.filters.model import filters
from pjimg.filters.value import *
