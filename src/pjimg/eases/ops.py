"""
Basic Usage: Eases
==================
The easing operation functions (eases) are used to change the values
within a set of image data. Using an easing operation (an "ease")
works like any other function.

Usage::

    >>> import numpy as np
    >>> a = np.array([[[0., .25, .5, .75, 1.], [0., .25, .5, .75, 1.]]])
    >>> in_circ(a)
    array([[[0.        , 0.03175416, 0.1339746 , 0.33856217, 1.        ],
            [0.        , 0.03175416, 0.1339746 , 0.33856217, 1.        ]]])

The functions themselves are simple. They all adhere to the Easing Operation
protocol:

    :param a: An array of image data.
    :return: The eased data as a :class:`numpy.ndarray`.
    :rtype: numpy.ndarray


Value Scaling
=============
Eases work best on image data. That is to say, they work best on arrays of
floating point data with values from zero to one. This is because the
math in many of the eases relies on the fact that multiplication results
in a smaller number and division results in a larger one.

What happens if you pass an array with numbers outside of that range?

If you pass an array with values less than zero or greater than one, the
:func:'pjimg.eases.will_scale' decorator will scale the data before
passing it to the ease and then unscale result before returning it.

What happens when the data is scaled?

When scaling the data, the function:

*   Sets the lowest value to zero,
*   Sets the highest value to one,
*   Sets the rest of the values to their proportional location between
    the lowest and the highest value.

It will then undo that scaling before returning the eased data.

This can result in some undesirable results if the lowest or highest value
in the data isn't the lowest or highest value in the value range you are
working in. For example, if you are working with the range 0–255 but the
lowest and highest values in the data are 127 and 215, the ease will
act like you are working in the range 127–215.

If this will cause problems for your code, you should scale the data
yourself before using the easing function.


Easing Operations
=================
The following are the easing functions available in :mod:`pjimg`.


Ease In
-------
These functions start slow or extend the darkness in an image:

.. autofunction:: pjimg.eases.in_back
.. autofunction:: pjimg.eases.in_bounce
.. autofunction:: pjimg.eases.in_circ
.. autofunction:: pjimg.eases.in_cubic
.. autofunction:: pjimg.eases.in_elastic
.. autofunction:: pjimg.eases.in_expo
.. autofunction:: pjimg.eases.in_quad
.. autofunction:: pjimg.eases.in_quart
.. autofunction:: pjimg.eases.in_quint
.. autofunction:: pjimg.eases.in_sin


Ease Out
--------
These functions start fast or extend the lightness in an image:

.. autofunction:: pjimg.eases.out_back
.. autofunction:: pjimg.eases.out_bounce
.. autofunction:: pjimg.eases.out_circ
.. autofunction:: pjimg.eases.out_cubic
.. autofunction:: pjimg.eases.out_elastic
.. autofunction:: pjimg.eases.out_quad
.. autofunction:: pjimg.eases.out_quint
.. autofunction:: pjimg.eases.out_sin


Ease In Out
-----------
These functions go fast in the middle or compress the midtones of the image.

.. autofunction:: pjimg.eases.in_out_back
.. autofunction:: pjimg.eases.in_out_circ
.. autofunction:: pjimg.eases.in_out_cos
.. autofunction:: pjimg.eases.in_out_cubic
.. autofunction:: pjimg.eases.in_out_elastic
.. autofunction:: pjimg.eases.in_out_perlin
.. autofunction:: pjimg.eases.in_out_quad
.. autofunction:: pjimg.eases.in_out_quint
.. autofunction:: pjimg.eases.in_out_sin


Ease Mid
--------
These functions change the values in the data, making the midtones dark
and the edges light.

.. autofunction:: pjimg.eases.mid_bump_linear
.. autofunction:: pjimg.eases.mid_bump_sin


Registration
============
All easing functions are registered in :class:`dict`
`pjimg.eases.eases` for convenience, but they can also
be called directly.
"""
import numpy as np

from pjimg.eases.decorators import register, will_scale
from pjimg.eases.model import Ease, eases
from pjimg.util import ImgAry


# Names available for import.
__all__ = [
    'in_back', 'in_bounce', 'in_circ', 'in_cubic', 'in_elastic',
    'in_expo',
    'in_out_back', 'in_out_circ', 'in_out_cos',
    'in_out_cubic', 'in_out_elastic', 'in_out_perlin',
    'in_out_quad',
    'in_out_quint', 'in_out_sin', 'in_quad',
    'in_quart',
    'in_quint', 'in_sin', 'mid_bump_linear',
    'mid_bump_sin', 'out_back', 'out_bounce', 'out_circ',
    'out_cubic', 'out_elastic', 'out_quad',
    'out_quint', 'out_sin',
]


# Ease in functions.
@register(eases)
@will_scale
def in_back(a: ImgAry) -> ImgAry:
    """An easing function that backs up a little before starting.
    
    .. figure:: images/plot_in_back.png
       :alt: A chart showing the action of the easing function.
       
       The action of :func:`in_back`.
       
    With image data, it extends the darker areas and compresses the
    lighter ones. The dip into negative values can be a little
    awkward. It's left to the calling application to decide how to
    handle it. In the following example, values are just truncated at
    zero.
    
    .. figure:: images/ex_in_back.png
       :alt: An example of the easing function affecting a gradient.
       
       An example of how :func:`in_back` affects a simple gradient. 
    
    :param a: An array of image data.
    :return: The eased data as a :class:`numpy.ndarray`.
    :rtype: numpy.ndarray
    """
    c1 = 1.70158
    c3 = c1 + 1
    return c3 * a ** 3 - c1 * a ** 2


@register(eases)
@will_scale
def in_bounce(a: ImgAry) -> ImgAry:
    """An easing function that has a bounce.
    
    .. figure:: images/plot_in_bounce.png
       :alt: A chart showing the action of the easing function.
       
       The action of :func:`in_bounce`.
       
    With image data, it is a large extension of the lighter areas
    with multiple peaks and compression of the darker ones.
    
    .. figure:: images/ex_in_bounce.png
       :alt: An example of the easing function affecting a gradient.
       
       An example of how :func:`in_bounce` affects a simple gradient. 
    
    :param a: An array of image data.
    :return: The eased data as a :class:`numpy.ndarray`.
    :rtype: numpy.ndarray
    """
    return 1 - out_bounce(1 - a)


@register(eases)
@will_scale
def in_circ(a: ImgAry) -> ImgAry:
    """An easing function that has a circular curve.
    
    .. figure:: images/plot_in_circ.png
       :alt: A chart showing the action of the easing function.
       
       The action of :func:`in_circ`.
       
    With image data, it is a moderate extension of the darker areas
    and compression of the lighter ones. This should not generate
    values outside of the original range.
    
    .. figure:: images/ex_in_circ.png
       :alt: An example of the easing function affecting a gradient.
       
       An example of how :func:`in_circ` affects a simple gradient. 
    
    :param a: An array of image data.
    :return: The eased data as a :class:`numpy.ndarray`.
    :rtype: numpy.ndarray
    """
    return 1 - np.sqrt(1 - a ** 2)


@register(eases)
@will_scale
def in_cubic(a: ImgAry) -> ImgAry:
    """An easing function that has a cubic curve.
    
    .. figure:: images/plot_in_cubic.png
       :alt: A chart showing the action of the easing function.
       
       The action of :func:`in_cubic`.
       
    With image data, it is a moderate extension of the darker areas
    and compression of the lighter ones. This should not generate
    values outside of the original range.
    
    .. figure:: images/ex_in_cubic.png
       :alt: An example of the easing function affecting a gradient.
       
       An example of how :func:`in_cubic` affects a simple gradient. 
    
    :param a: An array of image data.
    :return: The eased data as a :class:`numpy.ndarray`.
    :rtype: numpy.ndarray
    """
    return a ** 3


@register(eases)
@will_scale
def in_elastic(a: ImgAry) -> ImgAry:
    """An easing function that bounces.
    
    .. figure:: images/plot_in_elastic.png
       :alt: A chart showing the action of the easing function.
       
       The action of :func:`in_elastic`.
       
    With image data, it extends the darker areas and compresses the
    lighter ones. The dip into negative values can be a little
    awkward. It's left to the calling application to decide how to
    handle it. In the following example, values are just truncated at
    zero.
    
    .. figure:: images/ex_in_elastic.png
       :alt: An example of the easing function affecting a gradient.
       
       An example of how :func:`in_elastic` affects a simple gradient. 
    
    :param a: An array of image data.
    :return: The eased data as a :class:`numpy.ndarray`.
    :rtype: numpy.ndarray
    """
    c4 = (2 * np.pi) / 3
    m = np.zeros(a.shape, bool)
    m[a == 0] = True
    m[a == 1] = True
    a[~m] = -(2 ** (10 * a[~m] - 10)) * np.sin((a[~m] * 10 - 10.75) * c4)
    return a


@register(eases)
@will_scale
def in_expo(a: ImgAry) -> ImgAry:
    """An easing function that has an exponential curve.
    
    .. figure:: images/plot_in_expo.png
       :alt: A chart showing the action of the easing function.
       
       The action of :func:`in_expo`.
       
    With image data, it is a moderate extension of the darker areas
    and compression of the lighter ones. This should not generate
    values outside of the original range.
    
    .. figure:: images/ex_in_expo.png
       :alt: An example of the easing function affecting a gradient.
       
       An example of how :func:`in_expo` affects a simple gradient. 
    
    :param a: An array of image data.
    :return: The eased data as a :class:`numpy.ndarray`.
    :rtype: numpy.ndarray
    """
    a[a != 0] = 2 ** (10 * a[a !=0] - 10)
    return a


@register(eases)
@will_scale
def in_quad(a: ImgAry) -> ImgAry:
    """An easing function that has a quadratic curve.
    
    .. figure:: images/plot_in_quad.png
       :alt: A chart showing the action of the easing function.
       
       The action of :func:`in_quad`.
       
    With image data, it is a moderate extension of the darker areas
    and compression of the lighter ones. This should not generate
    values outside of the original range.
    
    .. figure:: images/ex_in_quad.png
       :alt: An example of the easing function affecting a gradient.
       
       An example of how :func:`in_quad` affects a simple gradient. 
    
    :param a: An array of image data.
    :return: The eased data as a :class:`numpy.ndarray`.
    :rtype: numpy.ndarray
    """
    return a ** 2


@register(eases)
@will_scale
def in_quart(a: ImgAry) -> ImgAry:
    """An easing function that has a quadric curve.
    
    .. figure:: images/plot_in_quart.png
       :alt: A chart showing the action of the easing function.
       
       The action of :func:`in_quart`.
       
    With image data, it is a moderate extension of the darker areas
    and compression of the lighter ones. This should not generate
    values outside of the original range.
    
    .. figure:: images/ex_in_quart.png
       :alt: An example of the easing function affecting a gradient.
       
       An example of how :func:`in_quart` affects a simple gradient. 
    
    :param a: An array of image data.
    :return: The eased data as a :class:`numpy.ndarray`.
    :rtype: numpy.ndarray
    """
    return a ** 4


@register(eases)
@will_scale
def in_quint(a: ImgAry) -> ImgAry:
    """An easing function that has a quintic curve.
    
    .. figure:: images/plot_in_quint.png
       :alt: A chart showing the action of the easing function.
       
       The action of :func:`in_quint`.
       
    With image data, it is a large extension of the darker areas
    and compression of the lighter ones. This should not generate
    values outside of the original range.
    
    .. figure:: images/ex_in_quint.png
       :alt: An example of the easing function affecting a gradient.
       
       An example of how :func:`in_quint` affects a simple gradient. 
    
    :param a: An array of image data.
    :return: The eased data as a :class:`numpy.ndarray`.
    :rtype: numpy.ndarray
    """
    return a ** 5


@register(eases)
@will_scale
def in_sin(a: ImgAry) -> ImgAry:
    """An easing function that has a sine curve.
    
    .. figure:: images/plot_in_sin.png
       :alt: A chart showing the action of the easing function.
       
       The action of :func:`in_sin`.
       
    With image data, it is a small extension of the darker areas
    and compression of the lighter ones. This should not generate
    values outside of the original range.
    
    .. figure:: images/ex_in_sin.png
       :alt: An example of the easing function affecting a gradient.
       
       An example of how :func:`in_sin` affects a simple gradient. 
    
    :param a: An array of image data.
    :return: The eased data as a :class:`numpy.ndarray`.
    :rtype: numpy.ndarray
    """
    return 1 - np.cos(a * np.pi / 2)


# Ease out functions.
@register(eases)
@will_scale
def out_back(a: ImgAry) -> ImgAry:
    """An easing function that backs up a little before ending.
    
    .. figure:: images/plot_out_back.png
       :alt: A chart showing the action of the easing function.
       
       The action of :func:`out_back`.
       
    With image data, it extends the lighter areas and compresses the
    darker ones. The dip into negative values can be a little
    awkward. It's left to the calling application to decide how to
    handle it. In the following example, values are just truncated at
    zero.
    
    .. figure:: images/ex_out_back.png
       :alt: An example of the easing function affecting a gradient.
       
       An example of how :func:`out_back` affects a simple gradient. 
    
    :param a: An array of image data.
    :return: The eased data as a :class:`numpy.ndarray`.
    :rtype: numpy.ndarray
    """
    c1 = 1.70158
    c3 = c1 + 1
    return 1 + c3 * (a - 1) ** 3 + c1 * (a - 1) ** 2


@register(eases)
@will_scale
def out_bounce(a: ImgAry) -> ImgAry:
    """An easing function that has a bounce.
    
    .. figure:: images/plot_out_bounce.png
       :alt: A chart showing the action of the easing function.
       
       The action of :func:`out_bounce`.
       
    With image data, it is a large extension of the lighter areas
    with multiple peaks and compression of the darker ones.
    
    .. figure:: images/ex_out_bounce.png
       :alt: An example of the easing function affecting a gradient.
       
       An example of how :func:`out_bounce` affects a simple gradient. 
    
    :param a: An array of image data.
    :return: The eased data as a :class:`numpy.ndarray`.
    :rtype: numpy.ndarray
    """
    n1 = 7.5625
    d1 = 2.75
    
    b = np.zeros(a.shape, dtype=a.dtype)

    b[a >= 2.5 / d1] = n1 * (a[a >= 2.5 / d1] - 2.625 / d1) ** 2 + .984375
    b[a < 2.5 / d1] = n1 * (a[a < 2.5 / d1] - 2.25 / d1) ** 2 + .9375
    b[a < 2 / d1] = n1 * (a[a < 2 / d1] - 1.5 / d1) ** 2 + .75
    b[a < 1 / d1] = n1 * a[a < 1 / d1] ** 2

    return b


@register(eases)
@will_scale
def out_circ(a: ImgAry) -> ImgAry:
    """An easing function that has a circular curve.
    
    .. figure:: images/plot_out_circ.png
       :alt: A chart showing the action of the easing function.
       
       The action of :func:`out_circ`.
       
    With image data, it is a moderate extension of the lighter areas
    and compression of the darker ones.
    
    .. figure:: images/ex_out_circ.png
       :alt: An example of the easing function affecting a gradient.
       
       An example of how :func:`out_circ` affects a simple gradient. 
    
    :param a: An array of image data.
    :return: The eased data as a :class:`numpy.ndarray`.
    :rtype: numpy.ndarray
    """
    return np.sqrt(1 - (a - 1) ** 2)


@register(eases)
@will_scale
def out_cubic(a: ImgAry) -> ImgAry:
    """An easing function that has a cubic curve.
    
    .. figure:: images/plot_out_cubic.png
       :alt: A chart showing the action of the easing function.
       
       The action of :func:`out_cubic`.
       
    With image data, it is a moderate extension of the lighter areas
    and compression of the darker ones.
    
    .. figure:: images/ex_out_cubic.png
       :alt: An example of the easing function affecting a gradient.
       
       An example of how :func:`out_cubic` affects a simple gradient. 
    
    :param a: An array of image data.
    :return: The eased data as a :class:`numpy.ndarray`.
    :rtype: numpy.ndarray
    """
    return 1 - (1 - a) ** 3


@register(eases)
@will_scale
def out_elastic(a: ImgAry) -> ImgAry:
    """An easing function that bounces.
    
    .. figure:: images/plot_out_elastic.png
       :alt: A chart showing the action of the easing function.
       
       The action of :func:`out_elastic`.
       
    With image data, it extends the lighter areas and compresses the
    darker ones. The bounce into values over one can be a little
    awkward. It's left to the calling application to decide how to
    handle it. In the following example, values are just truncated at
    one.
    
    .. figure:: images/ex_out_elastic.png
       :alt: An example of the easing function affecting a gradient.
       
       An example of how :func:`out_elastic` affects a simple gradient. 
    
    :param a: An array of image data.
    :return: The eased data as a :class:`numpy.ndarray`.
    :rtype: numpy.ndarray
    """
    c4 = (2 * np.pi) / 3
    m = np.zeros(a.shape, bool)
    m[a == 0] = True
    m[a == 1] = True
    a[~m] = 2 ** (-10 * a[~m]) * np.sin((a[~m] * 10 - .75) * c4) + 1
    return a


@register(eases)
@will_scale
def out_quad(a: ImgAry) -> ImgAry:
    """An easing function that has a quadratic curve.
    
    .. figure:: images/plot_out_quad.png
       :alt: A chart showing the action of the easing function.
       
       The action of :func:`out_quad`.
       
    With image data, it is a moderate extension of the lighter areas
    and compression of the darker ones.
    
    .. figure:: images/ex_out_quad.png
       :alt: An example of the easing function affecting a gradient.
       
       An example of how :func:`out_quad` affects a simple gradient. 
    
    :param a: An array of image data.
    :return: The eased data as a :class:`numpy.ndarray`.
    :rtype: numpy.ndarray
    """
    return 1 - (1 - a) ** 2


@register(eases)
@will_scale
def out_quint(a: ImgAry) -> ImgAry:
    """An easing function that has a quintic curve.
    
    .. figure:: images/plot_out_quint.png
       :alt: A chart showing the action of the easing function.
       
       The action of :func:`out_quint`.
       
    With image data, it is a large extension of the lighter areas
    and compression of the darker ones.
    
    .. figure:: images/ex_out_quint.png
       :alt: An example of the easing function affecting a gradient.
       
       An example of how :func:`out_quint` affects a simple gradient. 
    
    :param a: An array of image data.
    :return: The eased data as a :class:`numpy.ndarray`.
    :rtype: numpy.ndarray
    """
    return 1 - (1 - a) ** 5


@register(eases)
@will_scale
def out_sin(a: ImgAry) -> ImgAry:
    """An easing function that has a sine curve.
    
    .. figure:: images/plot_out_sin.png
       :alt: A chart showing the action of the easing function.
       
       The action of :func:`out_sin`.
       
    With image data, it is a small extension of the lighter areas
    and compression of the darker ones.
    
    .. figure:: images/ex_out_sin.png
       :alt: An example of the easing function affecting a gradient.
       
       An example of how :func:`out_sin` affects a simple gradient. 
    
    :param a: An array of image data.
    :return: The eased data as a :class:`numpy.ndarray`.
    :rtype: numpy.ndarray
    """
    return np.sin(a * np.pi / 2)


# Ease in and out functions.
@register(eases)
@will_scale
def in_out_back(a: ImgAry) -> ImgAry:
    """An easing function that backs up then overshoots.
    
    .. figure:: images/plot_in_out_back.png
       :alt: A chart showing the action of the easing function.
       
       The action of :func:`in_out_back`.
       
    With image data, it extends the darker and lighter areas. The dip
    into negative values and bounce over one can be a little awkward.
    It's left to the calling application to decide how to handle it. In
    the following example, values are just truncated at zero and one.
    
    .. figure:: images/ex_in_out_back.png
       :alt: An example of the easing function affecting a gradient.
       
       An example of how :func:`in_out_back` affects a simple gradient. 
    
    :param a: An array of image data.
    :return: The eased data as a :class:`numpy.ndarray`.
    :rtype: numpy.ndarray
    """
    c1 = 1.70158
    c2 = c1 * 1.525
    m = np.zeros(a.shape, bool)
    m[a < .5] = True
    a[m] = (2 * a[m]) ** 2 * ((c2 + 1) * 2 * a[m] - c2) / 2
    a[~m] = ((2 * a[~m] - 2) ** 2 * ((c2 + 1) * (a[~m] * 2 - 2) + c2) + 2) / 2
    return a


@register(eases)
@will_scale
def in_out_circ(a: ImgAry) -> ImgAry:
    """An easing function that uses a circular curve to compress the
    middle.
    
    .. figure:: images/plot_in_out_circ.png
       :alt: A chart showing the action of the easing function.
       
       The action of :func:`in_out_circ`.
       
    With image data, it extends the darker and lighter areas. This
    should not generate values outside of the original range.
    
    .. figure:: images/ex_in_out_circ.png
       :alt: An example of the easing function affecting a gradient.
       
       An example of how :func:`in_out_circ` affects a simple gradient. 
    
    :param a: An array of image data.
    :return: The eased data as a :class:`numpy.ndarray`.
    :rtype: numpy.ndarray
    """
    m = np.zeros(a.shape, bool)
    m[a < .5] = True
    a[m] = (1 - np.sqrt(1 - (2 * a[m]) ** 2)) / 2
    a[~m] = (np.sqrt(1 - (-2 * a[~m] + 2) ** 2) + 1) / 2
    return a


@register(eases)
@will_scale
def in_out_cos(a: ImgAry) -> ImgAry:
    """An easing function that uses a cosine curve to turn the make the
    middle low and the edges high.
    
    .. figure:: images/plot_in_out_cos.png
       :alt: A chart showing the action of the easing function.
       
       The action of :func:`in_out_cos`.
       
    With image data, it turns the midtones dark and the dark and light
    become midtones. This should not generate values outside of the
    original range.
    
    .. figure:: images/ex_in_out_cos.png
       :alt: An example of the easing function affecting a gradient.
       
       An example of how :func:`in_out_cos` affects a simple gradient. 
    
    :param a: An array of image data.
    :return: The eased data as a :class:`numpy.ndarray`.
    :rtype: numpy.ndarray
    """
    return -1 * (np.sin(np.pi * a) - 1) / 2


@register(eases)
@will_scale
def in_out_cubic(a: ImgAry) -> ImgAry:
    """An easing function that uses a cubic curve to compress the
    middle.
    
    .. figure:: images/plot_in_out_cubic.png
       :alt: A chart showing the action of the easing function.
       
       The action of :func:`in_out_cubic`.
       
    With image data, it extends the darker and lighter areas. This
    should not generate values outside of the original range.
    
    .. figure:: images/ex_in_out_cubic.png
       :alt: An example of the easing function affecting a gradient.
       
       An example of how :func:`in_out_cubic` affects a simple gradient. 
    
    :param a: An array of image data.
    :return: The eased data as a :class:`numpy.ndarray`.
    :rtype: numpy.ndarray
    """
    a[a < .5] = 4 * a[a < .5] ** 3
    a[a >= .5] = 1 - (-2 * a[a >= .5] + 2) ** 3 / 2
    return a


@register(eases)
@will_scale
def in_out_elastic(a: ImgAry) -> ImgAry:
    """An easing function that uses a bouncy curve to compress the
    middle.
    
    .. figure:: images/plot_in_out_elastic.png
       :alt: A chart showing the action of the easing function.
       
       The action of :func:`in_out_elastic`.
       
    With image data, it extends the darker and lighter areas. The 
    dip into values below zero or bounce into values over one can
    be a little awkward. It's left to the calling application to
    decide how to handle it. In the following example, values are
    just truncated at one.
    
    .. figure:: images/ex_in_out_elastic.png
       :alt: An example of the easing function affecting a gradient.
       
       An example of how :func:`in_out_elastic` affects a simple
       gradient. 
    
    :param a: An array of image data.
    :return: The eased data as a :class:`numpy.ndarray`.
    :rtype: numpy.ndarray
    """
    c5 = (2 * np.pi) / 4.5

    # Create masks for the array.
    m1 = np.zeros(a.shape, bool)
    m1[a < .5] = True
    m1[a <= 0] = False
    m2 = np.zeros(a.shape, bool)
    m2[a >= .5] = True
    m2[a >= 1] = False

    # Run the easing function based on the masks.
    a[m1] = -(2 ** (20 * a[m1] - 10) * np.sin((20 * a[m1] - 11.125) * c5))
    a[m1] = a[m1] / 2
    a[m2] = (2 ** (-20 * a[m2] + 10) * np.sin((20 * a[m2] - 11.125) * c5))
    a[m2] = a[m2] / 2 + 1
    return a


@register(eases)
@will_scale
def in_out_perlin(a: ImgAry) -> ImgAry:
    """An easing function that uses the easing equation from Ken
    Perlin's "Improved Perlin Noise" papaer.
    
    .. figure:: images/plot_in_out_perlin.png
       :alt: A chart showing the action of the easing function.
       
       The action of :func:`in_out_perlin`.
       
    With image data, it slightly extends the darker and lighter areas.
    This should not generate values outside of the original range.
    
    .. figure:: images/ex_in_out_perlin.png
       :alt: An example of the easing function affecting a gradient.
       
       An example of how :func:`in_out_perlin` affects a simple gradient. 
    
    :param a: An array of image data.
    :return: The eased data as a :class:`numpy.ndarray`.
    :rtype: numpy.ndarray
    """
    return 6 * a ** 5 - 15 * a ** 4 + 10 * a ** 3


@register(eases)
@will_scale
def in_out_quad(a: ImgAry) -> ImgAry:
    """An easing function that uses a quadratic curve to compress the
    middle.
    
    .. figure:: images/plot_in_out_quad.png
       :alt: A chart showing the action of the easing function.
       
       The action of :func:`in_out_quad`.
       
    With image data, it extends the darker and lighter areas. This
    should not generate values outside of the original range.
    
    .. figure:: images/ex_in_out_quad.png
       :alt: An example of the easing function affecting a gradient.
       
       An example of how :func:`in_out_quad` affects a simple
       gradient. 
    
    :param a: An array of image data.
    :return: The eased data as a :class:`numpy.ndarray`.
    :rtype: numpy.ndarray
    """
    m = np.zeros(a.shape, bool)
    m[a < .5] = True
    a[m] = 2 * a[m] ** 2
    a[~m] = 1 - (-2 * a[~m] + 2) ** 2 / 2
    return a


@register(eases)
@will_scale
def in_out_quint(a: ImgAry) -> ImgAry:
    """An easing function that uses a quintic curve to compress the
    middle.
    
    .. figure:: images/plot_in_out_quint.png
       :alt: A chart showing the action of the easing function.
       
       The action of :func:`in_out_quint`.
       
    With image data, it greatly extends the darker and lighter areas.
    This should not generate values outside of the original range.
    
    .. figure:: images/ex_in_out_quint.png
       :alt: An example of the easing function affecting a gradient.
       
       An example of how :func:`in_out_quint` affects a simple
       gradient. 
    
    :param a: An array of image data.
    :return: The eased data as a :class:`numpy.ndarray`.
    :rtype: numpy.ndarray
    """
    a[a < .5] = 16 * a[a < .5] ** 5
    a[a >= .5] = 1 - (-2 * a[a >= .5] + 2) ** 5 / 2
    return a


@register(eases)
@will_scale
def in_out_sin(a: ImgAry) -> ImgAry:
    """An easing function that uses a sine curve to compress the
    middle.
    
    .. figure:: images/plot_in_out_sin.png
       :alt: A chart showing the action of the easing function.
       
       The action of :func:`in_out_sin`.
       
    With image data, it slightly extends the darker and lighter areas.
    This should not generate values outside of the original range.
    
    .. figure:: images/ex_in_out_sin.png
       :alt: An example of the easing function affecting a gradient.
       
       An example of how :func:`in_out_sin` affects a simple gradient. 
    
    :param a: An array of image data.
    :return: The eased data as a :class:`numpy.ndarray`.
    :rtype: numpy.ndarray
    """
    return -1 * (np.cos(np.pi * a) - 1) / 2


# Ease mid functions.
@register(eases)
@will_scale
def mid_bump_linear(a: ImgAry) -> ImgAry:
    """An easing function that makes the middle of the range the peak
    of the values.
    
    .. figure:: images/plot_mid_bump_linear.png
       :alt: A chart showing the action of the easing function.
       
       The action of :func:`mid_bump_linear`.
       
    With image data, it makes the midtones light and the edges dark.
    
    .. figure:: images/ex_mid_bump_linear.png
       :alt: An example of the easing function affecting a gradient.
       
       An example of how :func:`mid_bump_linear` affects a simple
       gradient. 
    
    :param a: An array of image data.
    :return: The eased data as a :class:`numpy.ndarray`.
    :rtype: numpy.ndarray
    """
    a = np.abs(a - .5)
    m = np.zeros(a.shape, bool)
    m[a < .25] = True
    a[m] = (.25 - a[m]) * 4
    a[~m] = 0
    return a


@register(eases)
@will_scale
def mid_bump_sin(a: ImgAry) -> ImgAry:
    """An easing function that makes the middle of the range the peak
    of the values.
    
    .. figure:: images/plot_mid_bump_sin.png
       :alt: A chart showing the action of the easing function.
       
       The action of :func:`mid_bump_sin`.
       
    With image data, it makes the midtones light and the edges dark.
    
    .. figure:: images/ex_mid_bump_sin.png
       :alt: An example of the easing function affecting a gradient.
       
       An example of how :func:`mid_bump_sin` affects a simple
       gradient. 
    
    :param a: An array of image data.
    :return: The eased data as a :class:`numpy.ndarray`.
    :rtype: numpy.ndarray
    """
    a = np.abs(a - .5)
    m = np.zeros(a.shape, bool)
    m[a < .25] = True
    a[m] = (.25 - a[m]) * 4
    a[~m] = 0
    return in_out_sin(a)


if __name__ == '__main__':
    from pjimg.util.debug import print_array
    a = np.array([
        [
            [0.00, 0.25, 0.50, 0.75, 1.00, ],
            [0.25, 0.50, 0.75, 1.00, 0.75, ],
            [0.50, 0.75, 1.00, 0.75, 0.50, ],
            [0.75, 1.00, 0.75, 0.50, 0.25, ],
            [1.00, 0.75, 0.50, 0.25, 0.00, ],
        ],
    ], dtype=float)
    a = out_back(a)
    print_array(a)