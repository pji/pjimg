"""
Basic Usage: Blends
===================
The blending operation functions (blends) are used to blend two sets
of image data together. Using a blending operation (a "blend")
works like any other function.

Usage::

    >>> import numpy as np
    >>> a = np.array([[[0., .25, .5, .75, 1.], [0., .25, .5, .75, 1.]]])
    >>> b = np.array([[[1., 75, .5, .25, 0.], [1., 75, .5, .25, 0.]]])
    >>> darker(a, b)
    array([[[0.  , 0.25, 0.5 , 0.25, 0.  ],
            [0.  , 0.25, 0.5 , 0.25, 0.  ]]])

While the functions themselves are fairly simple, they are given some
extra functionality by decorators. Ultimately the true Blending
Operation protocol is:

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
handled through the :func:`pjimg.imgblend.will_match_size`
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

    >>> from pjimg.imgblend import will_colorize
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


Registration
============
All blending operations are registered in :class:`dict`
`pjimg.imgease.blends` for convenience, but they can also
be called directly.


Blend Operations
================

The examples in the following blends will show the result when a simple
horizontal gradient is blended with a simple vertical gradient.

    .. figure:: images/a.jpg
       :alt: The simple horizontal gradient `a`.
       
       The simple horizontal gradient `a`.
    
    .. figure:: images/b.jpg
       :alt: The simple horizontal gradient `b`.
       
       The simple horizontal gradient `b`.


Replacement Blends
------------------
.. autofunction:: pjimg.imgblend.replace


Darker/Burn Blends
------------------
.. autofunction:: pjimg.imgblend.darker
.. autofunction:: pjimg.imgblend.multiply
.. autofunction:: pjimg.imgblend.color_burn
.. autofunction:: pjimg.imgblend.linear_burn


Lighter/Dodge Blends
--------------------
.. autofunction:: pjimg.imgblend.lighter
.. autofunction:: pjimg.imgblend.screen
.. autofunction:: pjimg.imgblend.color_dodge
.. autofunction:: pjimg.imgblend.linear_dodge


Inversion Blends
----------------
.. autofunction:: pjimg.imgblend.difference
.. autofunction:: pjimg.imgblend.exclusion


Contrast Blends
---------------
.. autofunction:: pjimg.imgblend.hard_light
.. autofunction:: pjimg.imgblend.hard_mix
.. autofunction:: pjimg.imgblend.linear_light
.. autofunction:: pjimg.imgblend.overlay
.. autofunction:: pjimg.imgblend.pin_light
.. autofunction:: pjimg.imgblend.soft_light
.. autofunction:: pjimg.imgblend.vivid_light

"""
import numpy as np

from pjimg.imgblend.decorators import *
from pjimg.util import ImgAry


# Names available for export.
__all__ = [
    'replace', 'darker', 'multiply', 'color_burn', 'linear_burn',
    'lighter', 'screen', 'color_dodge', 'linear_dodge',
    'difference', 'exclusion', 'hard_light', 'hard_mix', 'linear_light',
    'overlay', 'pin_light', 'soft_light', 'vivid_light',
]


# Simple replacement blends.
@can_mask
@can_fade
@will_match_size
@will_colorize
def replace(a: ImgAry, b: ImgAry) -> ImgAry:
    """Simple replacement filter. Can double as an opacity filter
    if passed can_fade amount, but otherwise this will just replace the
    values in a with the values in b.

    .. figure:: images/replace.jpg
       :alt: The result of :func:`replace`.
       
       The result of :func:`replace`.

    :param a: The existing values. This is like the bottom layer in
        a photo editing tool.
    :param b: The values to blend. This is like the top layer in a
        photo editing tool.
    :param colorize: (Optional). Whether to ensure the two images have
        the same number of color channels.
    :param fade: (Optional.) The amount the blended values should
        affect the existing values. This is a float between zero
        and one, where zero is no effect and one is full effect.
        See :func:`imgblender.common.can_fade` for more details.
    :param mask: (Optional.) An image mask that is used to determine
        the effect the blend should have on the existing values.
        This is a :class:`numpy.ndarray` of floats between zero and
        one, where zero is no effect and one is full effect. See
        :func:`imgblender.common.can_mask` for details.
    :return: An :class:`numpy.ndarray` that contains the values of the
        blended arrays.
    :rtype: numpy.ndarray
    """
    return b


# Darker/burn blends.
@will_clip
@can_mask
@can_fade
@will_match_size
@will_colorize
def darker(a: ImgAry, b: ImgAry) -> ImgAry:
    """Replaces values in the existing image with values from the
    blending image when the value in the blending image is darker.

    .. figure:: images/darker.jpg
       :alt: The result of :func:`darker`.
       
       The result of :func:`darker`.

    :param a: The existing values. This is like the bottom layer in
        a photo editing tool.
    :param b: The values to blend. This is like the top layer in a
        photo editing tool.
    :param colorize: (Optional). Whether to ensure the two images have
        the same number of color channels.
    :param fade: (Optional.) The amount the blended values should
        affect the existing values. This is a float between zero
        and one, where zero is no effect and one is full effect.
        See :func:`imgblender.common.can_fade` for more details.
    :param mask: (Optional.) An image mask that is used to determine
        the effect the blend should have on the existing values.
        This is a :class:`numpy.ndarray` of floats between zero and
        one, where zero is no effect and one is full effect. See
        :func:`imgblender.common.can_mask` for details.
    :return: An :class:`numpy.ndarray` that contains the values of the
        blended arrays.
    :rtype: numpy.ndarray
    """
    ab = a.copy()
    ab[b < a] = b[b < a]
    return ab


@will_clip
@can_mask
@can_fade
@will_match_size
@will_colorize
def multiply(a: ImgAry, b: ImgAry) -> ImgAry:
    """Multiplies the values of the two images, leading to darker
    values. This is useful for shadows and similar situations.

    .. figure:: images/multiply.jpg
       :alt: The result of :func:`multiply`.
       
       The result of :func:`multiply`.

    :param a: The existing values. This is like the bottom layer in
        a photo editing tool.
    :param b: The values to blend. This is like the top layer in a
        photo editing tool.
    :param colorize: (Optional). Whether to ensure the two images have
        the same number of color channels.
    :param fade: (Optional.) The amount the blended values should
        affect the existing values. This is a float between zero
        and one, where zero is no effect and one is full effect.
        See :func:`imgblender.common.can_fade` for more details.
    :param mask: (Optional.) An image mask that is used to determine
        the effect the blend should have on the existing values.
        This is a :class:`numpy.ndarray` of floats between zero and
        one, where zero is no effect and one is full effect. See
        :func:`imgblender.common.can_mask` for details.
    :return: An :class:`numpy.ndarray` that contains the values of the
        blended arrays.
    :rtype: numpy.ndarray
    """
    return a * b


@will_clip
@can_mask
@can_fade
@will_match_size
@will_colorize
def color_burn(a: ImgAry, b: ImgAry) -> ImgAry:
    """Similar to multiply, but is darker and produces higher
    contrast.

    .. figure:: images/color_burn.jpg
       :alt: The result of :func:`color_burn`.
       
       The result of :func:`color_burn`.

    :param a: The existing values. This is like the bottom layer in
        a photo editing tool.
    :param b: The values to blend. This is like the top layer in a
        photo editing tool.
    :param colorize: (Optional). Whether to ensure the two images have
        the same number of color channels.
    :param fade: (Optional.) The amount the blended values should
        affect the existing values. This is a float between zero
        and one, where zero is no effect and one is full effect.
        See :func:`imgblender.common.can_fade` for more details.
    :param mask: (Optional.) An image mask that is used to determine
        the effect the blend should have on the existing values.
        This is a :class:`numpy.ndarray` of floats between zero and
        one, where zero is no effect and one is full effect. See
        :func:`imgblender.common.can_mask` for details.
    :return: An :class:`numpy.ndarray` that contains the values of the
        blended arrays.
    :rtype: numpy.ndarray
    """
    m = b != 0
    ab = np.zeros_like(a)
    ab[m] = 1 - (1 - a[m]) / b[m]
    ab[~m] = 0
    return ab


@will_clip
@can_mask
@can_fade
@will_match_size
@will_colorize
def linear_burn(a: ImgAry, b: ImgAry) -> ImgAry:
    """Similar to multiply, but is darker, produces less saturated
    colors than color burn, and produces more contrast in the shadows.

    .. figure:: images/linear_burn.jpg
       :alt: The result of :func:`linear_burn`.
       
       The result of :func:`linear_burn`.

    :param a: The existing values. This is like the bottom layer in
        a photo editing tool.
    :param b: The values to blend. This is like the top layer in a
        photo editing tool.
    :param colorize: (Optional). Whether to ensure the two images have
        the same number of color channels.
    :param fade: (Optional.) The amount the blended values should
        affect the existing values. This is a float between zero
        and one, where zero is no effect and one is full effect.
        See :func:`imgblender.common.can_fade` for more details.
    :param mask: (Optional.) An image mask that is used to determine
        the effect the blend should have on the existing values.
        This is a :class:`numpy.ndarray` of floats between zero and
        one, where zero is no effect and one is full effect. See
        :func:`imgblender.common.can_mask` for details.
    :return: An :class:`numpy.ndarray` that contains the values of the
        blended arrays.
    :rtype: numpy.ndarray
    """
    return a + b - 1


# Lighter/dodge blends.
@will_clip
@can_mask
@can_fade
@will_match_size
@will_colorize
def lighter(a: ImgAry, b: ImgAry) -> ImgAry:
    """Replaces values in the existing image with values from the
    blending image when the value in the blending image is lighter.

    .. figure:: images/lighter.jpg
       :alt: The result of :func:`lighter`.
       
       The result of :func:`lighter`.

    :param a: The existing values. This is like the bottom layer in
        a photo editing tool.
    :param b: The values to blend. This is like the top layer in a
        photo editing tool.
    :param colorize: (Optional). Whether to ensure the two images have
        the same number of color channels.
    :param fade: (Optional.) The amount the blended values should
        affect the existing values. This is a float between zero
        and one, where zero is no effect and one is full effect.
        See :func:`imgblender.common.can_fade` for more details.
    :param mask: (Optional.) An image mask that is used to determine
        the effect the blend should have on the existing values.
        This is a :class:`numpy.ndarray` of floats between zero and
        one, where zero is no effect and one is full effect. See
        :func:`imgblender.common.can_mask` for details.
    :return: An :class:`numpy.ndarray` that contains the values of the
        blended arrays.
    :rtype: numpy.ndarray
    """
    ab = a.copy()
    ab[b > a] = b[b > a]
    return ab


@will_clip
@can_mask
@can_fade
@will_match_size
@will_colorize
def screen(a: ImgAry, b: ImgAry) -> ImgAry:
    """Performs an inverse multiplication on the colors from the two
    images then inverse the colors again. This leads to overall
    brighter colors and is the opposite of multiply.

    .. figure:: images/screen.jpg
       :alt: The result of :func:`screen`.
       
       The result of :func:`screen`.

    :param a: The existing values. This is like the bottom layer in
        a photo editing tool.
    :param b: The values to blend. This is like the top layer in a
        photo editing tool.
    :param colorize: (Optional). Whether to ensure the two images have
        the same number of color channels.
    :param fade: (Optional.) The amount the blended values should
        affect the existing values. This is a float between zero
        and one, where zero is no effect and one is full effect.
        See :func:`imgblender.common.can_fade` for more details.
    :param mask: (Optional.) An image mask that is used to determine
        the effect the blend should have on the existing values.
        This is a :class:`numpy.ndarray` of floats between zero and
        one, where zero is no effect and one is full effect. See
        :func:`imgblender.common.can_mask` for details.
    :return: An :class:`numpy.ndarray` that contains the values of the
        blended arrays.
    :rtype: numpy.ndarray
    """
    rev_a = 1.0 - a
    rev_b = 1.0 - b
    ab = rev_a * rev_b
    return 1.0 - ab


@will_clip
@can_mask
@can_fade
@will_match_size
@will_colorize
def color_dodge(a: ImgAry, b: ImgAry) -> ImgAry:
    """Similar to screen, but brighter and decreases the contrast.

    .. figure:: images/color_dodge.jpg
       :alt: The result of :func:`color_dodge`.
       
       The result of :func:`color_dodge`.

    :param a: The existing values. This is like the bottom layer in
        a photo editing tool.
    :param b: The values to blend. This is like the top layer in a
        photo editing tool.
    :param colorize: (Optional). Whether to ensure the two images have
        the same number of color channels.
    :param fade: (Optional.) The amount the blended values should
        affect the existing values. This is a float between zero
        and one, where zero is no effect and one is full effect.
        See :func:`imgblender.common.can_fade` for more details.
    :param mask: (Optional.) An image mask that is used to determine
        the effect the blend should have on the existing values.
        This is a :class:`numpy.ndarray` of floats between zero and
        one, where zero is no effect and one is full effect. See
        :func:`imgblender.common.can_mask` for details.
    :return: An :class:`numpy.ndarray` that contains the values of the
        blended arrays.
    :rtype: numpy.ndarray
    """
    ab = np.ones_like(a)
    ab[b != 1] = a[b != 1] / (1 - b[b != 1])
    return ab


@will_clip
@can_mask
@can_fade
@will_match_size
@will_colorize
def linear_dodge(a: ImgAry, b: ImgAry) -> ImgAry:
    """Similar to screen but produces stronger results.

    .. figure:: images/linear_dodge.jpg
       :alt: The result of :func:`linear_dodge`.
       
       The result of :func:`linear_dodge`.

    :param a: The existing values. This is like the bottom layer in
        a photo editing tool.
    :param b: The values to blend. This is like the top layer in a
        photo editing tool.
    :param colorize: (Optional). Whether to ensure the two images have
        the same number of color channels.
    :param fade: (Optional.) The amount the blended values should
        affect the existing values. This is a float between zero
        and one, where zero is no effect and one is full effect.
        See :func:`imgblender.common.can_fade` for more details.
    :param mask: (Optional.) An image mask that is used to determine
        the effect the blend should have on the existing values.
        This is a :class:`numpy.ndarray` of floats between zero and
        one, where zero is no effect and one is full effect. See
        :func:`imgblender.common.can_mask` for details.
    :return: An :class:`numpy.ndarray` that contains the values of the
        blended arrays.
    :rtype: numpy.ndarray
    """
    return a + b


# Inversion blends.
@will_clip
@can_mask
@can_fade
@will_match_size
@will_colorize
def difference(a: ImgAry, b: ImgAry) -> ImgAry:
    """Takes the absolute value of the difference of the two values.
    This is often useful in creating complex patterns or when
    aligning two images.

    .. figure:: images/difference.jpg
       :alt: The result of :func:`difference`.
       
       The result of :func:`difference`.

    :param a: The existing values. This is like the bottom layer in
        a photo editing tool.
    :param b: The values to blend. This is like the top layer in a
        photo editing tool.
    :param colorize: (Optional). Whether to ensure the two images have
        the same number of color channels.
    :param fade: (Optional.) The amount the blended values should
        affect the existing values. This is a float between zero
        and one, where zero is no effect and one is full effect.
        See :func:`imgblender.common.can_fade` for more details.
    :param mask: (Optional.) An image mask that is used to determine
        the effect the blend should have on the existing values.
        This is a :class:`numpy.ndarray` of floats between zero and
        one, where zero is no effect and one is full effect. See
        :func:`imgblender.common.can_mask` for details.
    :return: An :class:`numpy.ndarray` that contains the values of the
        blended arrays.
    :rtype: numpy.ndarray
    """
    return np.abs(a - b)


@will_clip
@can_mask
@can_fade
@will_match_size
@will_colorize
def exclusion(a: ImgAry, b: ImgAry) -> ImgAry:
    """Similar to difference, with the result tending to gray
    rather than black.

    .. figure:: images/exclusion.jpg
       :alt: The result of :func:`exclusion`.
       
       The result of :func:`exclusion`.

    :param a: The existing values. This is like the bottom layer in
        a photo editing tool.
    :param b: The values to blend. This is like the top layer in a
        photo editing tool.
    :param colorize: (Optional). Whether to ensure the two images have
        the same number of color channels.
    :param fade: (Optional.) The amount the blended values should
        affect the existing values. This is a float between zero
        and one, where zero is no effect and one is full effect.
        See :func:`imgblender.common.can_fade` for more details.
    :param mask: (Optional.) An image mask that is used to determine
        the effect the blend should have on the existing values.
        This is a :class:`numpy.ndarray` of floats between zero and
        one, where zero is no effect and one is full effect. See
        :func:`imgblender.common.can_mask` for details.
    :return: An :class:`numpy.ndarray` that contains the values of the
        blended arrays.
    :rtype: numpy.ndarray
    """
    ab = a + b - 2 * a * b
    return ab


# Contrast blends.
@will_clip
@can_mask
@can_fade
@will_match_size
@will_colorize
def hard_light(a: ImgAry, b: ImgAry) -> ImgAry:
    """Similar to the blending image being a harsh light shining
    on the existing image.

    .. figure:: images/hard_light.jpg
       :alt: The result of :func:`hard_light`.
       
       The result of :func:`hard_light`.

    :param a: The existing values. This is like the bottom layer in
        a photo editing tool.
    :param b: The values to blend. This is like the top layer in a
        photo editing tool.
    :param colorize: (Optional). Whether to ensure the two images have
        the same number of color channels.
    :param fade: (Optional.) The amount the blended values should
        affect the existing values. This is a float between zero
        and one, where zero is no effect and one is full effect.
        See :func:`imgblender.common.can_fade` for more details.
    :param mask: (Optional.) An image mask that is used to determine
        the effect the blend should have on the existing values.
        This is a :class:`numpy.ndarray` of floats between zero and
        one, where zero is no effect and one is full effect. See
        :func:`imgblender.common.can_mask` for details.
    :return: An :class:`numpy.ndarray` that contains the values of the
        blended arrays.
    :rtype: numpy.ndarray
    """
    ab = np.zeros_like(a)
    ab[a < .5] = 2 * a[a < .5] * b[a < .5]
    ab[a >= .5] = 1 - 2 * (1 - a[a >= .5]) * (1 - b[a >= .5])
    return ab


@will_clip
@can_mask
@can_fade
@will_match_size
@will_colorize
def hard_mix(a: ImgAry, b: ImgAry) -> ImgAry:
    """Increases the saturation and contrast. It's best used with
    masks and can_fade.

    .. figure:: images/hard_mix.jpg
       :alt: The result of :func:`hard_mix`.
       
       The result of :func:`hard_mix`.

    :param a: The existing values. This is like the bottom layer in
        a photo editing tool.
    :param b: The values to blend. This is like the top layer in a
        photo editing tool.
    :param colorize: (Optional). Whether to ensure the two images have
        the same number of color channels.
    :param fade: (Optional.) The amount the blended values should
        affect the existing values. This is a float between zero
        and one, where zero is no effect and one is full effect.
        See :func:`imgblender.common.can_fade` for more details.
    :param mask: (Optional.) An image mask that is used to determine
        the effect the blend should have on the existing values.
        This is a :class:`numpy.ndarray` of floats between zero and
        one, where zero is no effect and one is full effect. See
        :func:`imgblender.common.can_mask` for details.
    :return: An :class:`numpy.ndarray` that contains the values of the
        blended arrays.
    :rtype: numpy.ndarray
    """
    ab = np.zeros_like(a)
    ab[a < 1 - b] = 0
    ab[a > 1 - b] = 1
    return ab


@will_clip
@can_mask
@can_fade
@will_match_size
@will_colorize
def linear_light(a: ImgAry, b: ImgAry) -> ImgAry:
    """Combines linear dodge and linear burn.

    .. figure:: images/linear_light.jpg
       :alt: The result of :func:`linear_light`.
       
       The result of :func:`linear_light`.

    :param a: The existing values. This is like the bottom layer in
        a photo editing tool.
    :param b: The values to blend. This is like the top layer in a
        photo editing tool.
    :param colorize: (Optional). Whether to ensure the two images have
        the same number of color channels.
    :param fade: (Optional.) The amount the blended values should
        affect the existing values. This is a float between zero
        and one, where zero is no effect and one is full effect.
        See :func:`imgblender.common.can_fade` for more details.
    :param mask: (Optional.) An image mask that is used to determine
        the effect the blend should have on the existing values.
        This is a :class:`numpy.ndarray` of floats between zero and
        one, where zero is no effect and one is full effect. See
        :func:`imgblender.common.can_mask` for details.
    :return: An :class:`numpy.ndarray` that contains the values of the
        blended arrays.
    :rtype: numpy.ndarray
    """
    ab = b + 2.0 * a - 1.0
    return ab


@will_clip
@can_mask
@can_fade
@will_match_size
@will_colorize
def overlay(a: ImgAry, b: ImgAry) -> ImgAry:
    """Combines screen and multiply blends.

    .. figure:: images/overlay.jpg
       :alt: The result of :func:`overlay`.
       
       The result of :func:`overlay`.

    :param a: The existing values. This is like the bottom layer in
        a photo editing tool.
    :param b: The values to blend. This is like the top layer in a
        photo editing tool.
    :param colorize: (Optional). Whether to ensure the two images have
        the same number of color channels.
    :param fade: (Optional.) The amount the blended values should
        affect the existing values. This is a float between zero
        and one, where zero is no effect and one is full effect.
        See :func:`imgblender.common.can_fade` for more details.
    :param mask: (Optional.) An image mask that is used to determine
        the effect the blend should have on the existing values.
        This is a :class:`numpy.ndarray` of floats between zero and
        one, where zero is no effect and one is full effect. See
        :func:`imgblender.common.can_mask` for details.
    :return: An :class:`numpy.ndarray` that contains the values of the
        blended arrays.
    :rtype: numpy.ndarray
    """
    mask = a >= .5
    ab = np.zeros_like(a)
    ab[~mask] = (2 * a[~mask] * b[~mask])
    ab[mask] = (1 - 2 * (1 - a[mask]) * (1 - b[mask]))
    return ab


@will_clip
@can_mask
@can_fade
@will_match_size
@will_colorize
def pin_light(a: ImgAry, b: ImgAry) -> ImgAry:
    """Combines lighten and darken blends.

    .. figure:: images/pin_light.jpg
       :alt: The result of :func:`pin_light`.
       
       The result of :func:`pin_light`.

    :param a: The existing values. This is like the bottom layer in
        a photo editing tool.
    :param b: The values to blend. This is like the top layer in a
        photo editing tool.
    :param colorize: (Optional). Whether to ensure the two images have
        the same number of color channels.
    :param fade: (Optional.) The amount the blended values should
        affect the existing values. This is a float between zero
        and one, where zero is no effect and one is full effect.
        See :func:`imgblender.common.can_fade` for more details.
    :param mask: (Optional.) An image mask that is used to determine
        the effect the blend should have on the existing values.
        This is a :class:`numpy.ndarray` of floats between zero and
        one, where zero is no effect and one is full effect. See
        :func:`imgblender.common.can_mask` for details.
    :return: An :class:`numpy.ndarray` that contains the values of the
        blended arrays.
    :rtype: numpy.ndarray
    """
    # Build array masks to handle how the algorithm changes.
    m1 = np.zeros(a.shape, bool)
    m1[b < 2 * a - 1] = True
    m2 = np.zeros(a.shape, bool)
    m2[b > 2 * a] = True
    m3 = np.zeros(a.shape, bool)
    m3[~m1] = True
    m3[m2] = False

    # Blend the arrays using the algorithm.
    ab = np.zeros_like(a)
    ab[m1] = 2 * a[m1] - 1
    ab[m2] = 2 * a[m2]
    ab[m3] = b[m3]
    return ab


@will_clip
@can_mask
@can_fade
@will_match_size
@will_colorize
def soft_light(a: ImgAry, b: ImgAry) -> ImgAry:
    """Similar to overlay, but biases towards the blending value
    rather than the existing value.

    .. figure:: images/soft_light.jpg
       :alt: The result of :func:`soft_light`.
       
       The result of :func:`soft_light`.

    :param a: The existing values. This is like the bottom layer in
        a photo editing tool.
    :param b: The values to blend. This is like the top layer in a
        photo editing tool.
    :param colorize: (Optional). Whether to ensure the two images have
        the same number of color channels.
    :param fade: (Optional.) The amount the blended values should
        affect the existing values. This is a float between zero
        and one, where zero is no effect and one is full effect.
        See :func:`imgblender.common.can_fade` for more details.
    :param mask: (Optional.) An image mask that is used to determine
        the effect the blend should have on the existing values.
        This is a :class:`numpy.ndarray` of floats between zero and
        one, where zero is no effect and one is full effect. See
        :func:`imgblender.common.can_mask` for details.
    :return: An :class:`numpy.ndarray` that contains the values of the
        blended arrays.
    :rtype: numpy.ndarray
    """
    m = np.zeros(a.shape, bool)
    ab = np.zeros_like(a)
    m[a < .5] = True
    ab[m] = (2 * a[m] - 1) * (b[m] - b[m] ** 2) + b[m]
    ab[~m] = (2 * a[~m] - 1) * (np.sqrt(b[~m]) - b[~m]) + b[~m]
    return ab


@will_clip
@can_mask
@can_fade
@will_match_size
@will_colorize
def vivid_light(a: ImgAry, b: ImgAry) -> ImgAry:
    """Good for color grading when faded.

    .. figure:: images/vivid_light.jpg
       :alt: The result of :func:`vivid_light`.
       
       The result of :func:`vivid_light`.

    :param a: The existing values. This is like the bottom layer in
        a photo editing tool.
    :param b: The values to blend. This is like the top layer in a
        photo editing tool.
    :param colorize: (Optional). Whether to ensure the two images have
        the same number of color channels.
    :param fade: (Optional.) The amount the blended values should
        affect the existing values. This is a float between zero
        and one, where zero is no effect and one is full effect.
        See :func:`imgblender.common.can_fade` for more details.
    :param mask: (Optional.) An image mask that is used to determine
        the effect the blend should have on the existing values.
        This is a :class:`numpy.ndarray` of floats between zero and
        one, where zero is no effect and one is full effect. See
        :func:`imgblender.common.can_mask` for details.
    :return: An :class:`numpy.ndarray` that contains the values of the
        blended arrays.
    :rtype: numpy.ndarray
    """
    # Create masks to handle the algorithm change and avoid division
    # by zero.
    m1 = np.zeros(a.shape, bool)
    m1[a <= .5] = True
    m1[a == 0] = False
    m2 = np.zeros(a.shape, bool)
    m2[a > .5] = True
    m2[a == 1] = False

    # Use the algorithm to blend the arrays.
    ab = np.zeros_like(a)
    ab[m1] = 1 - (1 - b[m1]) / (2 * a[m1])
    ab[m2] = b[m2] / (2 * (1 - a[m2]))
    return ab
