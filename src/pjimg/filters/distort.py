"""
Distortion Filters
------------------

Distortion filters are geometrical transformations that do not preserve
straight lines and parallels within in the image.

.. autofunction:: pjimg.filters.linear_to_polar
.. autofunction:: pjimg.filters.pinch
.. autofunction:: pjimg.filters.polar_to_linear
.. autofunction:: pjimg.filters.ripple
.. autofunction:: pjimg.filters.twirl

"""
from typing import Sequence

import cv2
import numpy as np
import skimage.transform as sktf  # type: ignore

from pjimg.filters.decorators import *
from pjimg.filters.model import filters
from pjimg.util import ImgAry, Loc, X, X_, Y, Y_, Z, Z_


# Names available for import.
__all__ = ['linear_to_polar', 'pinch', 'polar_to_linear', 'ripple', 'twirl',]

# Functions.
@register(filters)
@processes_by_grayscale_frame
@will_square
def linear_to_polar(a: ImgAry) -> ImgAry:
    """Convert the linear coordinates of the image data to
    polar coordinates.

    .. figure:: images/linear_to_polar.jpg
       :alt: An example of the filter affecting an image.
       
       An example of :func:`linear_to_polar` affecting an image.
    
    :param a: The image data to alter.
    :returns: A :class:`np.ndarray` object.
    :rtype: numpy.ndarray
    """
    center = tuple(n / 2 for n in a.shape)
    max_radius = np.sqrt(sum(n ** 2 for n in center))
    flags = cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP
    return cv2.warpPolar(a, a.shape, center, max_radius, flags)


@register(filters)
@processes_by_grayscale_frame
def pinch(
    a: ImgAry,
    amount: float,
    radius: float,
    scale: Sequence[float],
    offset: Loc = (0, 0, 0)
) -> ImgAry:
    """Distort an image to make it appear as though it is being
    pinched or swelling.

    .. figure:: images/pinch.jpg
       :alt: An example of the filter affecting an image.
       
       An example of :func:`pinch` affecting an image.
    
    :param a: The image data to alter.
    :param amount: How much the image should be distorted. Best results
        seem to be with numbers in the range of -1 <= x <= 1.
    :param radius: Sets the outside edge of the distortion, measured
        from the center of the distortion.
    :param scale: Adjusts the scale of the distortion. I'm not exactly
        clear on the math, but values less than one seem to increase
        the distortion. Values greater than one seem to decrease the
        distortion.
    :param offset: (Optional.) Sets how far the center of the
        distortion should be offset from the center of the image.
    :returns: A :class:`np.ndarray` object.
    :rtype: numpy.ndarray
    
    .. warning::
        If done too close to the edge of the image data, you will get
        artifacts due to the lack of data. To calculate the minimum
        safe distance from the edge:
        
            radius * (1 + amount)
    """
    # Set up for creating the maps.
    center = tuple((n) / 2 + o for n, o in zip(a.shape, offset))

    # Create a map of the distance from each pixel in the image to
    # the center of the image.
    indices = np.indices(a.shape)
    y = indices[Y_]
    x = indices[X_]
    delta_y = scale[Y_] * (y - center[Y_])
    delta_x = scale[X_] * (x - center[X_])
    distance = delta_x ** 2 + delta_y ** 2

    # Mask out the area covered by not within the radius of the effect.
    r_mask = np.zeros(x.shape, bool)
    r_mask[distance >= radius ** 2] = True
    flex_x = np.zeros(a.shape, np.float32)
    flex_y = np.zeros(a.shape, np.float32)
    flex_x[r_mask] = x[r_mask]
    flex_y[r_mask] = y[r_mask]

    # Create maps with the barrel/pincushion formula.
    pmask = np.zeros(x.shape, bool)
    pmask[distance > 0.0] = True
    pmask[r_mask] = False
    factor = np.sin(np.pi * np.sqrt(distance) / radius / 2)
    factor[factor > 0] = factor[factor > 0] ** -amount
    factor[factor < 0] = -((-factor[factor < 0]) ** -amount)
    flex_x[pmask] = factor[pmask] * delta_x[pmask] / scale[X_] + center[X_]
    flex_y[pmask] = factor[pmask] * delta_y[pmask] / scale[Y_] + center[Y_]

    flex_x[~pmask] = 1.0 * delta_x[~pmask] / scale[X_] + center[X_]
    flex_y[~pmask] = 1.0 * delta_y[~pmask] / scale[Y_] + center[Y_]

    # Perform the pinch using the maps and return.
    return cv2.remap(a, flex_x, flex_y, cv2.INTER_LINEAR)


@register(filters)
@processes_by_grayscale_frame
@will_square
def polar_to_linear(a: ImgAry) -> ImgAry:
    """Convert the polar coordinates of the image data to
    linear coordinates.

    .. figure:: images/polar_to_linear.jpg
       :alt: An example of the filter affecting an image.
       
       An example of :func:`polar_to_linear` affecting an image.
    
    :param a: The image data to alter.
    :returns: A :class:`np.ndarray` object.
    :rtype: numpy.ndarray
    """
    center = tuple(n / 2 for n in a.shape)
    max_radius = np.sqrt(sum(n ** 2 for n in center))
    return cv2.linearPolar(a, center, max_radius, cv2.WARP_FILL_OUTLIERS)


@register(filters)
@processes_by_grayscale_frame
def ripple(
    a: ImgAry,
    wave: Sequence[float],
    amp: Sequence[float],
    distaxis: Sequence[int],
    offset: Loc = (0, 0, 0)
) -> ImgAry:
    """Perform a ripple distortion.

    .. figure:: images/ripple.jpg
       :alt: An example of the filter affecting an image.
       
       An example of :func:`ripple` affecting an image.
    
    :param a: The image data to alter.
    :param wave: The distance between peaks in the distortion.
        There needs to be one value in the sequence per dimension
        in the image.
    :param amp: The amount of change caused by each ripple. There
        needs to be one value in the sequence per dimension in the
        image.
    :param distaxis: Whether the distortion should be along the
        same axis being distorted, causing the pattern to bunch up
        like it is rippling, or along a different axis, causing the
        pattern to wave like it's the cross-section of a wave. The
        values are the indexes of the axis to distort along.
    :param offset: (Optional.) The amount to offset the location
        of the ripples in the image. There needs to be one value
        in the sequence per dimension in the image. The default
        value for all dimensions is zero.
    :returns: A :class:`np.ndarray` object.
    :rtype: numpy.ndarray
    """
    # Map out the volume of the given image and make sure everything is
    # in float32 to keep the cv2.remap function happy.
    flex = np.indices(a.shape, np.float32)
    flex_x = flex[X_].copy()
    flex_y = flex[Y_].copy()

    # Modify the mapping to apply the ripple to create the flex
    # maps for cv.remap. The flex map value for each pixel will
    # indicate how far that pixel moves in the remapped image.
    *_, da_x, da_y = distaxis
    *_, off_y, off_x = offset
    if wave[X_]:
        flex_x = np.cos((off_x + flex[da_x]) / wave[X_] * 2 * np.pi)
        flex_x = flex[X_] + flex_x * amp[X_]
    if wave[Y_]:
        flex_y = np.cos((off_y + flex[da_y]) / wave[Y_] * 2 * np.pi)
        flex_y = flex[Y_] + flex_y * amp[Y_]

    # Remap the color values in the original image using the
    # rippled flex map.
    return cv2.remap(a, flex_x, flex_y, cv2.INTER_LINEAR)


@register(filters)
@processes_by_grayscale_frame
def twirl(
    a: ImgAry,
    radius: float,
    strength: float,
    offset: tuple[int, int] = (0, 0)
) -> ImgAry:
    """Swirl the image data.

    .. figure:: images/twirl.jpg
       :alt: An example of the filter affecting an image.
       
       An example of :func:`twirl` affecting an image.
    
    :param a: The image data to alter.
    :param radius: The location of the edge of the distortion. This
        is measured from the center of the distortion.
    :param strength: The amount of the distortion. Its roughly
        equivalent to the number of rotations the distortion makes
        around the center of the distortion.
    :param offset: (Optional.) How far to offset the center of the
        distortion from the center of the image.
    :returns: A :class:`np.ndarray` object.
    :rtype: numpy.ndarray
    """
    # Determine the location of the center of the twirl effect.
    center = [n / 2 + o for n, o in zip(a.shape, offset)]
    return sktf.swirl(a, center[::-1], strength, radius)
