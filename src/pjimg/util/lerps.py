"""
lerps
~~~~~

Interpolation functions.

.. autofunction:: pjimg.util.cubic_interpolation
.. autofunction:: pjimg.util.linear_interpolation
.. autofunction:: pjimg.util.n_dimensional_interpolation
.. autofunction:: pjimg.util.n_dimensional_cubic_interpolation
.. autofunction:: pjimg.util.n_dimensional_linear_interpolation
"""
from functools import partial
from math import prod
from typing import Callable, Optional

import numpy as np

from pjimg.util.decorators import preserves_type
from pjimg.util.model import Interpolator, NumAry, RatioAry


# Importable names.
__all__ = [
    'cerp', 'cubic_interpolation', 'lerp', 'linear_interpolation',
    'n_dimensional_cubic_interpolation', 'n_dimensional_interpolation',
    'n_dimensional_linear_interpolation', 'nderp', 'ndcerp', 'ndlerp',
]


# Public interpolation functions.
@preserves_type
def cubic_interpolation(
    a: NumAry,
    b: NumAry,
    x: RatioAry,
    a_: Optional[NumAry] = None,
    b_: Optional[NumAry] = None
) -> NumAry:
    """Perform a cubic interpolation on the values of four arrays.
    This is adapted from code found at: `Cubic Interpolation`_

    .. _`Cubic Interpolation`: https://www.paulinternet.nl/?page=bicubic

    :param a: The closest value on the "left" side.
    :param b: The closest value on the "right" side.
    :param x: How close the final value is to the closest "left" value.
    :param a_: (Optional.) The farther value on the "left" side.
    :param b_: (Optional.) The farther value on the "right" side.
    :return: A :class:`numpy.ndarray` object.
    :rtype: numpy.ndarray

    Usage::

        >>> import numpy as np
        >>>
        >>> base = np.array([0, 1, 2, 3, 4, 5], dtype=float)
        >>> a = base[1:4]
        >>> b = base[2:5]
        >>> a_ = base[:3]
        >>> b_ = base[3:]
        >>> x = np.array([0.5, 0.5, 0.5])
        >>> cubic_interpolation(a, b, x, a_, b_)
        array([1.5, 2.5, 3.5])

    You can do the interpolation if you don't have the furthest data
    points. It will just be inaccurate at the edges because of the
    missing data::

        >>> base = np.array([1, 2, 3, 4], dtype=float)
        >>> a = base[:3]
        >>> b = base[1:]
        >>> x = np.array([0.5, 0.5, 0.5])
        >>> cubic_interpolation(a, b, x)
        array([1.4375, 2.5   , 3.5625])

    """
    # Cubic interpolation needs two points to the left of the
    # interpolation spot and two points to the right. If only two
    # points were given, assume they are the points on the closest
    # side of the interpolation and create arrays for the furthest
    # points using the given points.
    #
    # Note: At the edges, this guesses the values by repeating values.
    # This isn't as accurate as passing in better values with a_
    # and b_.
    if a_ is None:
        a_ = np.roll(a, 1, -1)
        a_[..., 0] = a[..., 0]
    if b_ is None:
        b_ = np.roll(b, -1, -1)
        b_[..., -1] = b[..., -1]

    # Perform the interpolation. This is broken up to keep it within
    # the 80 character width limit.
    part1 = (3 * (a - b) + b_ - a_)
    part2 = (2 * a_ - 5 * a + 4 * b - b_ + x * part1)
    return a + 0.5 * x * (b - a_ + x * part2)


@preserves_type
def linear_interpolation(
    a: NumAry,
    b: NumAry,
    x: RatioAry,
) -> NumAry:
    """Perform a linear interpolation on the values of two arrays

    :param a: The "left" values. The datatype of a also determines the
        datatype of the returned array.
    :param b: The "right" values.
    :param x: An array of how close the location of the final value
        should be to the "left" value.
    :return: A :class:`numpy.ndarray` object.
    :rtype: numpy.ndarray

    Usage::

        >>> import numpy as np
        >>>
        >>> a = np.array([1, 2, 3])
        >>> b = np.array([3, 4, 5])
        >>> x = np.array([.5, .5, .5])
        >>> linear_interpolation(a, b, x)
        array([2, 3, 4])
    """
    return a * (1 - x) + b * x


def n_dimensional_interpolation(
    a: NumAry,
    b: NumAry,
    x: RatioAry,
    interpolator: Interpolator
) -> NumAry:
    """Perform an interpolation over multiple dimensions.

    :param a: The "left" values.
    :param b: The "right" values.
    :param x: An array of how close the location of the final value
        should be to the "left" value.
    :return: A :class:`numpy.ndarray` object.
    :rtype: numpy.ndarray

    Usage::

        >>> import numpy as np
        >>>
        >>> a = np.zeros((2, 3, 3), dtype=int)
        >>> b = np.full((2, 3, 3), 255, dtype=int)
        >>> x = np.linspace(0.0, 1.0, 18, True, False, float)
        >>> x = x.reshape((2, 3, 3))
        >>> n_dimensional_interpolation(a, b, x, lerp)
        array([[135, 150, 165],
               [179, 194, 210],
               [225, 240, 255]])
    """
    # N-dimensional interpolation uses the nearest points to make a
    # reasonable guess at the value of a point between them. The
    # number of points used is proportional to the number of
    # dimensions. This will do a quick check to make sure enough
    # points were supplied for the interpolation.
    if len(a) + len(b) != 2 ** len(x):
        msg = 'Not the correct number of points for the dimensions.'
        raise ValueError(msg)

    # Recursively interpolate the points.
    if len(x) > 1:
        interpolated = interpolator(a, b, x[-1])
        a = interpolated[::2]
        b = interpolated[1::2]
        return n_dimensional_interpolation(a, b, x[:-1], interpolator)

    # The extra dimension in the result is caused by the extra
    # dimension in a, b, and x to hold the arrays that will be
    # interpolated. The only way to avoid it would be to iterate
    # through a, b, and x rather than just doing the math over all
    # of them at once, but that would be much slower.
    result = interpolator(a, b, x[0])
    return result[0]


def n_dimensional_cubic_interpolation(
    a: NumAry,
    b: NumAry,
    x: RatioAry,
) -> NumAry:
    """Perform a cubic interpolation over multiple dimensions. This
    is a shortcut for :func:`lerpy.n_dimensional_interpolation` using
    :func:`cubic_interpolation` as an interpolator.

    :param a: The "left" values. The datatype of a also determines the
        datatype of the returned array.
    :param b: The "right" values.
    :param x: An array of how close the location of the final value
        should be to the "left" value.
    :return: A :class:`numpy.ndarray` object.
    :rtype: numpy.ndarray
    """
    return n_dimensional_interpolation(a, b, x, cubic_interpolation)


def n_dimensional_linear_interpolation(
    a: NumAry,
    b: NumAry,
    x: RatioAry,
) -> NumAry:
    """Perform a linear interpolation over multiple dimensions. This
    is a shortcut for :func:`lerpy.n_dimensional_interpolation` using
    :func:`linear_interpolation` as an interpolator.

    :param a: The "left" values. The datatype of a also determines the
        datatype of the returned array.
    :param b: The "right" values.
    :param x: An array of how close the location of the final value
        should be to the "left" value.
    :return: A :class:`numpy.ndarray` object.
    :rtype: numpy.ndarray
    """
    return n_dimensional_interpolation(a, b, x, linear_interpolation)


# Function aliases.
cerp = cubic_interpolation
lerp = linear_interpolation
nderp = n_dimensional_interpolation
ndcerp = n_dimensional_cubic_interpolation
ndlerp = n_dimensional_linear_interpolation

