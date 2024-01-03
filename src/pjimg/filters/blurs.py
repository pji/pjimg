"""
Blur Filters
------------

Blur filters alter the image by blending the values of different pixels
within the image.

.. autofunction:: pjimg.filters.box_blur
.. autofunction:: pjimg.filters.gaussian_blur
.. autofunction:: pjimg.filters.glow
.. autofunction:: pjimg.filters.motion_blur

"""
import cv2
import numpy as np

from pjimg.filters.decorators import *
from pjimg.filters.filters_ import filters
from pjimg.util import ImgAry, X, X_, Y, Y_, Z, Z_


@register(filters)
@processes_by_grayscale_frame
def box_blur(a: ImgAry, size: int) -> ImgAry:
    """Perform a box blur.

    .. figure:: images/box_blur.jpg
       :alt: An example of the filter affecting an image.
       
       An example of :func:`box_blur` affecting an image.
    
    :param a: The image data to alter.
    :param size: The size of the blox used in the blur.
    :returns: A :class:`np.ndarray` object.
    :rtype: numpy.ndarray
    """
    kernel = np.ones((size, size), float) / size ** 2
    return cv2.filter2D(a, -1, kernel)


@register(filters)
@processes_by_grayscale_frame
def gaussian_blur(a: ImgAry, sigma: float) -> ImgAry:
    """Perform a gaussian blur.

    .. figure:: images/gaussian_blur.jpg
       :alt: An example of the filter affecting an image.
       
       An example of :func:`gaussian_blur` affecting an image.
    
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


@register(filters)
def glow(a: ImgAry, sigma: int) -> ImgAry:
    """Use gaussian blurs to create a halo around brighter objects
    in the image.

    .. figure:: images/glow.jpg
       :alt: An example of the filter affecting an image.
       
       An example of :func:`glow` affecting an image.
    
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
        b = gaussian_blur(b, sigma)
        b = _screen(a, b)
        sigma = sigma // 2
    return b


@register(filters)
@processes_by_grayscale_frame
def motion_blur(
    a: ImgAry,
    amount: int,
    axis: int
) -> ImgAry:
    """Perform a motion blur.

    .. figure:: images/motion_blur.jpg
       :alt: An example of the filter affecting an image.
       
       An example of :func:`motion_blur` affecting an image.
    
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
