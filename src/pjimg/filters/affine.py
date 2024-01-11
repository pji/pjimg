"""
Affine Transformation Filters
-----------------------------

Affine transformation filters are geometrical transformations that
preserve straight lines and parallels within the image.

.. autofunction:: pjimg.filters.flip
.. autofunction:: pjimg.filters.grow
.. autofunction:: pjimg.filters.rotate_2d
.. autofunction:: pjimg.filters.rotate_90
.. autofunction:: pjimg.filters.skew

"""
from typing import Optional

import cv2
import numpy as np

import pjimg.util.resize as rsz
from pjimg.filters.decorators import *
from pjimg.filters.model import filters
from pjimg.util import find_center, ImgAry, Loc, X, X_, Y, Y_, Z, Z_


# Names available for import.
__all__ = [
    'flip', 'grow', 'rotate_90', 'rotate_2d', 'skew',
]


# Functions.
@register(filters)
def flip(a: ImgAry, axis: int) -> ImgAry:
    """Flip the image around an axis.

    .. figure:: images/flip.jpg
       :alt: An example of the filter affecting an image.
       
       An example of :func:`flip` affecting an image.
    
    :param a: The image data to alter.
    :param axis: The axis to flip the image data around.
    :returns: A :class:`np.ndarray` object.
    :rtype: numpy.ndarray
    """
    return np.flip(a, axis)


@register(filters)
def grow(a: ImgAry, factor: float, yx_only: bool = False) -> ImgAry:
    """Increase the size of an image.

    .. figure:: images/grow.jpg
       :alt: An example of the filter affecting an image.
       
       An example of :func:`grow` affecting an image.
    
    :param a: The image data to alter.
    :param factor: The scaling factor to use when increasing the
        size of the image.
    :param xy_only: (Optional.) Only grow the length and the width
        of a three-dimensional array.
    :returns: A :class:`np.ndarray` object.
    :rtype: numpy.ndarray
    """
    if yx_only and len(a.shape) > 2:
        frames = [grow(frame, factor) for frame in a]
        return np.array(frames)
    
    if len(a.shape) == 2:
        return rsz.bilinear_interpolation(a, factor)
    return rsz.trilinear_interpolation(a, factor)


@register(filters)
@processes_by_grayscale_frame
def rotate_2d(
    a: ImgAry, angle: float, origin: Optional[Loc] = None, safe: bool = True
) -> ImgAry:
    """Rotate the image by an arbitrary angle around the Z axis.
    
    .. figure:: images/rotate_2d.jpg
       :alt: An example of the filter affecting an image.
       
       An example of :func:`rotate_2d` affecting an image.
    
    :param a: The image data to alter.
    :param angle: The angle to rotate the image in degrees.
    :param origin: (Optional.) The point of rotation. Defaults to
        the center of the image.
    :param safe: (Optional.) Make a defensive copy of the image data
        before operating on the data to prevent unexpected changes to
        the original image data. This can be turned off in cases where
        it is more important to preserve the memory. Defaults to `True`.
    :returns: An array of image data.
    :rtype: A :class:numpy.ndarray object.
    """
    if safe:
        a = a.copy()
    if origin is None:
        origin = find_center(a.shape)
    y, x = origin
    matrix = cv2.getRotationMatrix2D((x, y), angle, 1)
    return cv2.warpAffine(a, matrix, (a.shape[X_], a.shape[Y_]))


@register(filters)
def rotate_90(a: ImgAry, direction: str = 'cw') -> ImgAry:
    """Rotate the data 90° around the Z axis.

    .. figure:: images/rotate_90.jpg
       :alt: An example of the filter affecting an image.
       
       An example of :func:`rotate_90` affecting an image.
    
    :param a: The image data to alter.
    :param direction: (Optional.) Whether to rotate the data
        clockwise or counter clockwise.
    :returns: An array of image data.
    :rtype: A :class:numpy.ndarray object.
    """
    spin = -1
    if direction in ['ccw', 'counter clockwise', 'l', 'left']:
        spin = 1
    return np.rot90(a, spin, (Y, X))


@register(filters)
@processes_by_grayscale_frame
def skew(a: ImgAry, slope: float) -> ImgAry:
    """Perform a skew distort on the data.

    .. figure:: images/skew.jpg
       :alt: An example of the filter affecting an image.
       
       An example of :func:`skew` affecting an image.
    
    :param a: The image data to alter.
    :param slope: The slope of the Y axis of the image after the skew.
    :returns: A :class:`np.ndarray` object.
    :rtype: numpy.ndarray
    """
    # Create the transform matrix by defining three points in the
    # original image, and then showing where they move to in the
    # new, transformed image. The order of the axes is reversed
    # for this in comparison to how it's generally used in pjinoise.
    # This is due to the implementation of OpenCV.
    original = np.array([
        [0, 0],
        [a.shape[X_] - 1, 0],
        [0, a.shape[Y_] - 1],
    ], dtype=np.float32)
    new = np.array([
        [0, 0],
        [a.shape[X_] - 1, 0],
        [(a.shape[Y_] - 1) * slope, a.shape[Y_] - 1],
    ], dtype=np.float32)

    # Perform the transform on the image by first creating a warp
    # matrix from the example points. Then apply that matrix to
    # the image, telling OpenCV to wrap pixels that are pushed off
    # the edge of the image.
    matrix = cv2.getAffineTransform(original, new)
    return cv2.warpAffine(
        a, matrix, (a.shape[X_], a.shape[Y_]), borderMode=cv2.BORDER_WRAP
    )
