"""
Value Filters
=============
Value filters operate on the values of the image data without any
geometrical transformations or blurs. They are somewhat similar to
the easing functions found in :mod:``

.. autofunction:: pjimg.filters.colorize
.. autofunction:: pjimg.filters.contrast
.. autofunction:: pjimg.filters.cut_highlight
.. autofunction:: pjimg.filters.cut_shadow
.. autofunction:: pjimg.filters.distance
.. autofunction:: pjimg.filters.inverse
.. autofunction:: pjimg.filters.posterize

"""
from typing import Optional, Sequence

import cv2
import numpy as np
import skimage.transform as sktf  # type: ignore
from PIL import Image, ImageOps

from pjimg.filters.decorators import *
from pjimg.filters.model import Filter, filters
from pjimg.filters.util import get_color_for_key
from pjimg.util import find_center, ImgAry, Loc, Size, X, Y, Z, X_, Y_, Z_


# Names available for import.
__all__ = [
    'colorize', 'contrast', 'cut_highlight', 'cut_shadow', 'distance',
    'filters', 'inverse', 'posterize',
]


# Image filter functions.
@register(filters)
@processes_by_grayscale_frame
@uses_uint8
def colorize(
    a: ImgAry,
    colorkey: str = '',
    white: str = '#FFFFFF',
    black: str = '#000000'
) -> ImgAry:
    """Colorize a grayscale image.

    .. figure:: images/colorize.jpg
       :alt: An example of the filter affecting an image.
   
       An example of :func:`colorize` affecting an image.

    :param a: The image data to alter.
    :param colorkey: (Optional.) The key for the pre-defined
        colors to use in the colorization. These are defined
        in `pjimg.filters.constants.COLORS`.
    :param white: (Optional.) The color name for the color
        to use to replace white in the image. Color names
        are defined by PIL.ImageColor.
    :param black: (Optional.) The color name for the color
        to use to replace black in the image. Color names
        are defined by PIL.ImageColor.
    :returns: A :class:`np.ndarray` object.
    :rtype: numpy.ndarray

    .. warning::
        The output of this filter is in RGB color rather than
        grayscale. This will impact the ability to use the
        output with other features of :mod:`pjimg`.
    
    """
    src_space = 'L'
    dst_space = 'RGB'
    if colorkey:
        white, black = get_color_for_key(colorkey)
    img = Image.fromarray(a, mode=src_space)
    colorized = ImageOps.colorize(
        image=img,
        black=black,
        white=white,
        blackpoint=0x00,
        midpoint=0x7f,
        whitepoint=0xff
    )
    img = colorized.convert(dst_space)
    out = np.array(img, dtype=a.dtype)
    return out


@register(filters)
def contrast(
    a: ImgAry, black: float = 0.0, white: float = 1.0
) -> ImgAry:
    """Adjust the image to fill the full dynamic range.

    .. figure:: images/contrast.jpg
       :alt: An example of the filter affecting an image.
       
       An example of :func:`contrast` affecting an image.
    
    :param a: The image data to alter.
    :param black: (Optional.) The minimum value in the output.
    :param white: (Optional.) The maximum value in the output.
    :returns: A :class:`np.ndarray` object.
    :rtype: numpy.ndarray
    """
    # Normalize the values to a scale from 0.0 to 1.0.
    a_min = np.min(a)
    a_max = np.max(a)
    scale = a_max - a_min
    if scale != 0:
        a = a - a_min
        a = a / scale
    else:
        a.fill(0.5)
    
    # Scale to the destination range.
    dest_scale = white - black
    if dest_scale != 1.0:
        a = a * dest_scale
        a += black
    return a


@register(filters)
def cut_highlight(a: ImgAry, threshold: float) -> ImgAry:
    """Set the white point of the image to the given threshold, removing
    detail from and increasing the size of the highlights.
    
    .. figure:: images/cut_highlight.jpg
       :alt: An example of the filter affecting an image.
       
       An example of :func:`cut_highlight` affecting an image.
    
    :param a: The image data to alter.
    :param threshold: The threshold value for the new white point.
    :returns: A :class:`np.ndarray` object.
    :rtype: numpy.ndarray
    """
    a[a > threshold] = threshold
    result = a / threshold
    return result.astype(np.float64)


@register(filters)
def cut_shadow(a: ImgAry, threshold: float) -> ImgAry:
    """Set the black point of the image to the given threshold, removing
    detail from and increasing the size of the shadows.
    
    .. figure:: images/cut_shadow.jpg
       :alt: An example of the filter affecting an image.
       
       An example of :func:`cut_shadow` affecting an image.
    
    :param a: The image data to alter.
    :param threshold: The threshold value for the new black point.
    :returns: A :class:`np.ndarray` object.
    :rtype: numpy.ndarray
    """
    a = 1.0 - a
    threshold = 1.0 - threshold
    a[a > threshold] = threshold
    result = a / threshold
    return 1.0 - result


@register(filters)
@processes_by_grayscale_frame
@uses_uint8
def distance(a: ImgAry, mask_size: int = 5) -> ImgAry:
    """Make each pixel the distance to the nearest black pixel in the
    original image.
    
    .. figure:: images/distance.jpg
       :alt: An example of the filter affecting an image.
       
       An example of :func:`distance` affecting an image.
    
    :param a: The image data to alter.
    :param mask_size: (Optional.) The size of the mask used to determine
        the distance to a black pixel.
    :returns: A :class:`np.ndarray` object.
    :rtype: numpy.ndarray
    """
    dist_type = cv2.DIST_L2
    result = cv2.distanceTransform(a, dist_type, mask_size)
    return (result / np.max(result)) * 0xff


@register(filters)
def inverse(a: ImgAry) -> ImgAry:
    """Inverse the colors of an image.

    .. figure:: images/inverse.jpg
       :alt: An example of the filter affecting an image.
       
       An example of :func:`inverse` affecting an image.
    
    :param a: The image data to alter.
    :returns: A :class:`np.ndarray` object.
    :rtype: numpy.ndarray
    """
    return 1 - a


@register(filters)
def posterize(a: ImgAry, levels: int = 2):
    """Reduce the number of colors in the image data.
    
    .. figure:: images/posterize.jpg
       :alt: An example of the filter affecting an image.
       
       An example of :func:`posterize` affecting an image.
    
    :param a: The image data to alter.
    :param levels: (Optional.) The number of colors in the resulting data.
        Default is 2.
    :returns: A :class:`np.ndarray` object.
    :rtype: numpy.ndarray
    """
    a = a.copy()
    a *= levels - 1
    a = np.around(a, 0)
    a /= levels - 1
    return a.astype(float)


if __name__ == '__main__':
    from pjimg.util.debug import print_array
    
    a = np.array([
        [0.00, 0.25, 0.50, 0.75, 1.00,],
        [0.25, 0.50, 0.75, 1.00, 0.75,],
        [0.50, 0.75, 1.00, 0.75, 0.50,],
        [0.75, 1.00, 0.75, 0.50, 0.25,],
        [1.00, 0.75, 0.50, 0.25, 0.00,],
    ], dtype=float)
    v = np.array([
        [
            [0.00, 0.25, 0.50, 0.75, 1.00,],
            [0.25, 0.50, 0.75, 1.00, 0.75,],
            [0.50, 0.75, 1.00, 0.75, 0.50,],
            [0.75, 1.00, 0.75, 0.50, 0.25,],
            [1.00, 0.75, 0.50, 0.25, 0.00,],
        ],
        [
            [1.00, 0.75, 0.50, 0.25, 0.00,],
            [0.75, 1.00, 0.75, 0.50, 0.25,],
            [0.50, 0.75, 1.00, 0.75, 0.50,],
            [0.25, 0.50, 0.75, 1.00, 0.75,],
            [0.00, 0.25, 0.50, 0.75, 1.00,],
        ],
    ], dtype=float)

    a = distance(a)
    print_array(a)
