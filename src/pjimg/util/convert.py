"""
convert
~~~~~~~

Data conversion utilities.

.. autofunction:: pjimg.util.float_to_uint8
.. autofunction:: pjimg.util.grayscale_to_rgb

"""
import numpy as np

from pjimg.util.model import ArrayLike, ImgAry, IntAry


# Exported names.
__all__ = ['float_to_uint8', 'grayscale_to_rgb',]


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


def grayscale_to_rgb(a: ImgAry) -> ImgAry:
    """Convert single channel image data to three channel.
    
    :param a: The array of grayscale image data to convert.
    :return: A :class:`numpy.ndarray` object.
    :rtype: numpy.ndarray
    """
    new_shape = (*a.shape, 3)
    new_a = np.zeros(new_shape, dtype=a.dtype)
    for channel in range(3):
        new_a[..., channel] = a
    return new_a
