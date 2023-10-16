"""
Basic Usage: Filters
====================
The filter operation functions (filters) are used to make changes to
values in image data where the resulting value of each pixel can be
influenced by the values of other pixels in the data.

Usage::

    >>> import numpy as np
    >>> a = np.array([[[0., .25, .5, .75, 1.], [0., .25, .5, .75, 1.]]])
    >>> filter_box_blur(a, size=2)
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

.. autofunction:: pjimg.filters.filter_box_blur
.. autofunction:: pjimg.filters.filter_colorize
.. autofunction:: pjimg.filters.filter_contrast
.. autofunction:: pjimg.filters.filter_flip
.. autofunction:: pjimg.filters.filter_gaussian_blur
.. autofunction:: pjimg.filters.filter_glow
.. autofunction:: pjimg.filters.filter_grow
.. autofunction:: pjimg.filters.filter_inverse
.. autofunction:: pjimg.filters.filter_linear_to_polar
.. autofunction:: pjimg.filters.filter_motion_blur
.. autofunction:: pjimg.filters.filter_pinch
.. autofunction:: pjimg.filters.filter_polar_to_linear
.. autofunction:: pjimg.filters.filter_ripple
.. autofunction:: pjimg.filters.filter_rotate_90
.. autofunction:: pjimg.filters.filter_skew
.. autofunction:: pjimg.filters.filter_twirl

"""
from typing import Sequence

import cv2
import numpy as np
import skimage.transform as sktf  # type: ignore
from PIL import Image, ImageOps

import pjimg.util.resize as rsz
from pjimg.filters.decorators import *
from pjimg.filters.util import get_color_for_key
from pjimg.util import ImgAry, Loc, Size, X, Y, Z, X_, Y_, Z_


# Names available for import.
__all__ = [
    'filter_box_blur', 'filter_colorize', 'filter_contrast',
    'filter_flip', 'filter_gaussian_blur', 'filter_glow',
    'filter_grow', 'filter_inverse', 'filter_linear_to_polar',
    'filter_motion_blur', 'filter_pinch', 'filter_polar_to_linear',
    'filter_ripple', 'filter_rotate_90', 'filter_skew',
    'filter_twirl',
]


# Image filter functions.
@processes_by_grayscale_frame
def filter_box_blur(a: ImgAry, size: int) -> ImgAry:
    """Perform a box blur.

    .. figure:: images/filter_box_blur.jpg
       :alt: An example of the filter affecting an image.
       
       An example of :func:`filter_box_blur` affecting an image.
    
    :param a: The image data to alter.
    :param size: The size of the blox used in the blur.
    :returns: A :class:`np.ndarray` object.
    :rtype: numpy.ndarray
    """
    kernel = np.ones((size, size), float) / size ** 2
    return cv2.filter2D(a, -1, kernel)


@processes_by_grayscale_frame
@uses_uint8
def filter_colorize(
    a: ImgAry,
    colorkey: str = '',
    white: str = '#FFFFFF',
    black: str = '#000000'
) -> ImgAry:
    """Colorize a grayscale image.

    .. figure:: images/filter_colorize.jpg
       :alt: An example of the filter affecting an image.
       
       An example of :func:`filter_colorize` affecting an image.
    
    :param a: The image data to alter.
    :param colorkey: (Optional.) The key for the pre-defined
        colors to use in the colorization. These are defined
        in utility.COLOR.
    :param white: (Optional.) The color name for the color
        to use to replace white in the image. Color names
        are defined by PIL.ImageColor.
    :param black: (Optional.) The color name for the color
        to use to replace black in the image. Color names
        are defined by PIL.ImageColor.
    :returns: A :class:`np.ndarray` object.
    :rtype: numpy.ndarray
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


def filter_contrast(
    a: ImgAry, black: float = 0.0, white: float = 1.0
) -> ImgAry:
    """Adjust the image to fill the full dynamic range.

    .. figure:: images/filter_contrast.jpg
       :alt: An example of the filter affecting an image.
       
       An example of :func:`filter_contrast` affecting an image.
    
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


def filter_flip(a: ImgAry, axis: int) -> ImgAry:
    """Flip the image around an axis.

    .. figure:: images/filter_flip.jpg
       :alt: An example of the filter affecting an image.
       
       An example of :func:`filter_flip` affecting an image.
    
    :param a: The image data to alter.
    :param axis: The axis to flip the image data around.
    :returns: A :class:`np.ndarray` object.
    :rtype: numpy.ndarray
    """
    return np.flip(a, axis)


@processes_by_grayscale_frame
def filter_gaussian_blur(a: ImgAry, sigma: float) -> ImgAry:
    """Perform a gaussian blur.

    .. figure:: images/filter_gaussian_blur.jpg
       :alt: An example of the filter affecting an image.
       
       An example of :func:`filter_gaussian_blur` affecting an image.
    
    :param a: The image data to alter.
    :param sigma: The sigma value of the blur. A gaussian blur uses a
        gaussian function to determine how much the other pixels in
        the image should affect the value of a pixel. Gaussian
        functions produce a normal distribution. This value is the
        size of a standard deviation in that normal distribution.
    :returns: A :class:`numpy.ndarray` object.
    :rtype: numpy.ndarray
    """
    return cv2.GaussianBlur(a, (0, 0), sigma, sigma, 0)


def filter_glow(a: ImgAry, sigma: int) -> ImgAry:
    """Use gaussian blurs to create a halo around brighter objects
    in the image.

    .. figure:: images/filter_glow.jpg
       :alt: An example of the filter affecting an image.
       
       An example of :func:`filter_gaussian_blur` affecting an image.
    
    :param a: The image data to alter.
    :param sigma: The sigma value of the blur. A gaussian blur uses a
        gaussian function to determine how much the other pixels in
        the image should affect the value of a pixel. Gaussian
        functions produce a normal distribution. This value is the
        size of a standard deviation in that normal distribution.
    :returns: A :class:`np.ndarray` object.
    :rtype: numpy.ndarray
    """
    def _screen(a: ImgAry, b: ImgAry) -> ImgAry:
        rev_a = 1 - a
        rev_b = 1 - b
        ab = rev_a * rev_b
        return 1.0 - ab

    b = a.copy()
    while sigma > 0:
        if sigma % 2 != 1:
            sigma -= 1
        b = filter_gaussian_blur(b, sigma)
        b = _screen(a, b)
        sigma = sigma // 2
    return b


def filter_grow(a: ImgAry, factor: float) -> ImgAry:
    """Increase the size of an image.

    .. figure:: images/filter_grow.jpg
       :alt: An example of the filter affecting an image.
       
       An example of :func:`filter_grow` affecting an image.
    
    :param a: The image data to alter.
    :param factor: The scaling factor to use when increasing the
        size of the image.
    :returns: A :class:`np.ndarray` object.
    :rtype: numpy.ndarray
    """
    if len(a.shape) == 2:
        return rsz.bilinear_interpolation(a, factor)
    return rsz.trilinear_interpolation(a, factor)


def filter_inverse(a: ImgAry) -> ImgAry:
    """Inverse the colors of an image.

    .. figure:: images/filter_inverse.jpg
       :alt: An example of the filter affecting an image.
       
       An example of :func:`filter_inverse` affecting an image.
    
    :param a: The image data to alter.
    :returns: A :class:`np.ndarray` object.
    :rtype: numpy.ndarray
    """
    return 1 - a


@processes_by_grayscale_frame
@will_square
def filter_linear_to_polar(a: ImgAry) -> ImgAry:
    """Convert the linear coordinates of the image data to
    polar coordinates.

    .. figure:: images/filter_linear_to_polar.jpg
       :alt: An example of the filter affecting an image.
       
       An example of :func:`filter_linear_to_polar` affecting an image.
    
    :param a: The image data to alter.
    :returns: A :class:`np.ndarray` object.
    :rtype: numpy.ndarray
    """
    center = tuple(n / 2 for n in a.shape)
    max_radius = np.sqrt(sum(n ** 2 for n in center))
    flags = cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP
    return cv2.warpPolar(a, a.shape, center, max_radius, flags)


@processes_by_grayscale_frame
def filter_motion_blur(
    a: ImgAry,
    amount: int,
    axis: int
) -> ImgAry:
    """Perform a motion blur.

    .. figure:: images/filter_motion_blur.jpg
       :alt: An example of the filter affecting an image.
       
       An example of :func:`filter_motion_blur` affecting an image.
    
    :param a: The image data to alter.
    :param size: The size of the blur to apply.
    :param direction: The axis that the blur should be performed along.
        The index should be indicated using the filters.X or filters.Y
        objects.
    :returns: A :class:`np.ndarray` object.
    :rtype: numpy.ndarray
    """
    kernel = np.zeros((amount, amount), float)
    if axis == X_:
        y = int(amount // 2)
        for x in range(amount):
            kernel[y][x] = 1 / amount
    elif axis == Y_:
        x = int(amount // 2)
        for y in range(amount):
            kernel[y][x] = 1 / amount
    else:
        raise ValueError('motion_blur can only affect the X or Y axis.')
    return cv2.filter2D(a, -1, kernel)


@processes_by_grayscale_frame
def filter_pinch(
    a: ImgAry,
    amount: float,
    radius: float,
    scale: Sequence[float],
    offset: Loc = (0, 0, 0)
) -> ImgAry:
    """Distort an image to make it appear as though it is being
    pinched or swelling.

    .. figure:: images/filter_pinch.jpg
       :alt: An example of the filter affecting an image.
       
       An example of :func:`filter_pinch` affecting an image.
    
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
    """
    # Set up for creating the maps.
    center = tuple((n) / 2 + o for n, o in zip(a.shape, offset))
    flex_x = np.zeros(a.shape, np.float32)
    flex_y = np.zeros(a.shape, np.float32)

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


@processes_by_grayscale_frame
@will_square
def filter_polar_to_linear(a: ImgAry) -> ImgAry:
    """Convert the polar coordinates of the image data to
    linear coordinates.

    .. figure:: images/filter_polar_to_linear.jpg
       :alt: An example of the filter affecting an image.
       
       An example of :func:`filter_polar_to_linear` affecting an image.
    
    :param a: The image data to alter.
    :returns: A :class:`np.ndarray` object.
    :rtype: numpy.ndarray
    """
    center = tuple(n / 2 for n in a.shape)
    max_radius = np.sqrt(sum(n ** 2 for n in center))
    return cv2.linearPolar(a, center, max_radius, cv2.WARP_FILL_OUTLIERS)


@processes_by_grayscale_frame
def filter_ripple(
    a: ImgAry,
    wave: Sequence[float],
    amp: Sequence[float],
    distaxis: Sequence[int],
    offset: Loc = (0, 0, 0)
) -> ImgAry:
    """Perform a ripple distortion.

    .. figure:: images/filter_ripple.jpg
       :alt: An example of the filter affecting an image.
       
       An example of :func:`filter_ripple` affecting an image.
    
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


def filter_rotate_90(a: ImgAry, direction: str = 'cw') -> ImgAry:
    """Rotate the data 90° around the Z axis.

    .. figure:: images/filter_rotate_90.jpg
       :alt: An example of the filter affecting an image.
       
       An example of :func:`filter_filter_rotate_90` affecting an image.
    
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


@processes_by_grayscale_frame
def filter_skew(a: ImgAry, slope: float) -> ImgAry:
    """Perform a skew distort on the data.

    .. figure:: images/filter_skew.jpg
       :alt: An example of the filter affecting an image.
       
       An example of :func:`filter_skew` affecting an image.
    
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


@processes_by_grayscale_frame
def filter_twirl(
    a: ImgAry,
    radius: float,
    strength: float,
    offset: tuple[int, int] = (0, 0)
) -> ImgAry:
    """Swirl the image data.

    .. figure:: images/filter_twirl.jpg
       :alt: An example of the filter affecting an image.
       
       An example of :func:`filter_twirl` affecting an image.
    
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
