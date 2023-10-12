"""
imgblend
~~~~~~~~

Blending operations to use when combining two sets of image data.

Many of these are taken from:

    *   http://www.deepskycolors.com/archive/2010/04/21/
        formulas-for-Photoshop-blending-modes.html
    *   http://www.simplefilter.de/en/basics/mixmods.html


Basic Usage: Blends
===================
The blending operation functions (blends) are used to blend two sets
of image data together. Using a blending operation (an "operation")
works like any other function all. The parameters follow the Blending
Operation protocol.

Usage::

    >>> import numpy as np
    >>> a = np.array([[[0., .25, .5, .75, 1.], [0., .25, .5, .75, 1.]]])
    >>> b = np.array([[[1., 75, .5, .25, 0.], [1., 75, .5, .25, 0.]]])
    >>> darker(a, b)
    array([[[0.  , 0.25, 0.5 , 0.25, 0.  ],
            [0.  , 0.25, 0.5 , 0.25, 0.  ]]])

While the functions themselves are fairly simple, they are given some
extra functionality by decorators. Ultimately the true protocol for the
operations is:

    :param a: The image data from the existing image.
    :param b: The image data from the blending image.
    :param colorize: (Optional). Whether to ensure the two images have
        the same number of color channels.
    :param fade: (Optional.) (From @can_fade.) How much the blend
        should impact the final output. This is a percentage, so the
        range of valid values are 0 <= x <= 1.
    :param mask: (Optional.) (From @mcan_mask.) An array of data used
        to mask the blending operation. This is also a percentage, so a
        value of one in the mask means that pixel is fully affected by
        the operation. A value of zero means the pixel is not affected
        by the operation.
    :return: A :class:`numpy.ndarray` object.
    :rtype: numpy.ndarray


Colorize, Array Shape, and Color Channels
=========================================
The blends themselves don't care about the dimensionality of the given
arrays. It just needs the two arrays to have the same shape by the
time it does the blend. While this was originally written for image
data, to the algorithms themselves, it's all just floating-point math.

However, there is one case where a bias towards image data shows up:

    * You pass two arrays with differing shapes,
    * The size of their last dimension is different,
    * One of the two arrays has a last dimension with size three.

To perform the blending algorithm, the two arrays must be the same
shape. In most cases, differences between the two shapes will be
handled through the :func:`imgblender.common.will_match_size`
decorator, which adds zeros to the smaller array to make their sizes
match. However, in the case described above, something different
happens.

Since color image data often has a last dimension size of three,
representing color channels, the case above is intercepted by the
:func:`img.blender.common.will_colorize` decorator. That decorator
assumes the array that doesn't have a last dimension size of three
is single channel image data ("grayscale") and will add a new last
dimension of size three. The values will be the original single
value repeated three times. To demonstrate::

    >>> from imgblender.common import will_colorize
    >>> a = np.array([
    ...     [1.0, 0.5, 0.0, ],
    ...     [0.5, 0.0, 0.5, ],
    ...     [0.0, 0.5, 1.0, ],
    ... ])
    >>> b = np.array([
    ...     [[0, 0, 0], [0, 0, 0], [0, 0, 0], ],
    ...     [[0, 0, 0], [0, 0, 0], [0, 0, 0], ],
    ...     [[0, 0, 0], [0, 0, 0], [0, 0, 0], ],
    ... ])
    >>>
    >>> @will_colorize
    ... def spam(a, b):
    ...     return a
    ...
    >>> a_ = spam(a, b)
    >>> a_
    array([[[1. , 1. , 1. ],
            [0.5, 0.5, 0.5],
            [0. , 0. , 0. ]],
    <BLANKLINE>
           [[0.5, 0.5, 0.5],
            [0. , 0. , 0. ],
            [0.5, 0.5, 0.5]],
    <BLANKLINE>
           [[0. , 0. , 0. ],
            [0.5, 0.5, 0.5],
            [1. , 1. , 1. ]]])
    >>> a.shape
    (3, 3)
    >>> a_.shape
    (3, 3, 3)

The value returned by ```spam()``` in the demonstration has an
extra dimension of size three added, and the values are three copies
of the values in the original ```a````.

This can be turned off by passing ```False``` to the ```colorize```
parameter of the blend.


.. automodule:: pjimg.imgblend.ops
.. automodule:: pjimg.imgblend.decorators
"""
from pjimg.imgblend.decorators import *
from pjimg.imgblend.model import Blend
from pjimg.imgblend.ops import *
