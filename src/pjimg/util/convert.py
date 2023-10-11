"""
convert
~~~~~~~

Data conversion utilities.

.. autofunction:: pjimg.util.float_to_uint8
"""
import numpy as np

from pjimg.util.model import ArrayLike, IntAry


# Exported names.
__all__ = ['float_to_uint8',]


# Functions.
def float_to_uint8(a: ArrayLike) -> IntAry:
    """Convert an array of floating point values to an array of
    unsigned 8-bit integers.

    :param a: The array of image data to convert to unsigned 8-bit
        integers.
    :return: A :class:`numpy.ndarray` object.
    :rtype: numpy.ndarray
    """
    a = np.array(a)
    if np.max(a) > 1 or np.min(a) < 0:
        msg = 'Array values must be 0 >= x >= 1.'
        raise ValueError(msg)
    a *= 0xff
    return a.astype(np.uint8)
