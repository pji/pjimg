"""
Basic Usage: Filters
====================
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


Registration
============
All filter functions are registered in the :class:`dict`
`pjimg.filters.filters` for convenience, but they can also
be called directly.


Filter Functions
================
The following are the filter functions available in :mod:`pjimg`.

.. autofunction:: pjimg.filters.colorize
.. autofunction:: pjimg.filters.contrast
.. autofunction:: pjimg.filters.cut_highlight
.. autofunction:: pjimg.filters.cut_shadow
.. autofunction:: pjimg.filters.flip
.. autofunction:: pjimg.filters.grow
.. autofunction:: pjimg.filters.inverse
.. autofunction:: pjimg.filters.linear_to_polar
.. autofunction:: pjimg.filters.pinch
.. autofunction:: pjimg.filters.polar_to_linear
.. autofunction:: pjimg.filters.posterize
.. autofunction:: pjimg.filters.ripple
.. autofunction:: pjimg.filters.rotate_2d
.. autofunction:: pjimg.filters.rotate_90
.. autofunction:: pjimg.filters.skew
.. autofunction:: pjimg.filters.twirl

"""
from typing import Optional, Sequence

import cv2
import numpy as np
import skimage.transform as sktf  # type: ignore
from PIL import Image, ImageOps

import pjimg.util.resize as rsz
from pjimg.filters.decorators import *
from pjimg.filters.model import Filter
from pjimg.filters.util import get_color_for_key
from pjimg.util import find_center, ImgAry, Loc, Size, X, Y, Z, X_, Y_, Z_


# Names available for import.
__all__ = [
    'colorize', 'contrast', 'cut_highlight', 'cut_shadow',
    'filters', 'flip',
    'grow', 'inverse', 'linear_to_polar',
    'pinch', 'polar_to_linear', 'posterize',
    'ripple', 'rotate_2d', 'rotate_90', 'skew',
    'twirl'
]


# Registry of filter functions.
filters: dict[str, Filter] = dict()


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
    a = a / threshold
    return a


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
    a = a / threshold
    return 1.0 - a


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
    
    ..warning:
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
    size = a.shape
    if len(size) > 2:
        for i, frame in enumerate(a):
            frame = rotate_2d(frame, angle, origin, safe)
            a[i] = frame[np.newaxis, :, :]
        return a
    
    if origin is None:
        origin = find_center(size)
    y, x = origin
    matrix = cv2.getRotationMatrix2D((x, y), angle, 1)
    return cv2.warpAffine(a, matrix, (size[X_], size[Y_]))


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

    a = rotate_2d(a, 45, (1, 1))
    print_array(a)
