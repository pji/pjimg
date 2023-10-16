"""
Decorators
==========

Common decorators used by :mod:`pjimg.blends`.

.. autofunction:: pjimg.blends.can_fade
.. autofunction:: pjimg.blends.can_mask
.. autofunction:: pjimg.blends.will_clip
.. autofunction:: pjimg.blends.will_colorize
.. autofunction:: pjimg.blends.will_match_size

"""
from functools import wraps
from typing import Callable, Union

import numpy as np

from pjimg.blends.model import Blend
from pjimg.util import ImgAry, grayscale_to_rgb, pad_array


# Names available for import.
__all__ = [
    'can_fade', 'can_mask', 'will_clip', 'will_colorize', 'will_match_size'
]


# Decorators.
def can_fade(fn: Blend) -> Blend:
    """Adjust how much the blend affects the base array."""
    @wraps(fn)
    def wrapper(
        a: ImgAry,
        b: ImgAry,
        fade: float = 1.0,
        *args, **kwargs
    ) -> ImgAry:
        # Get the blended image from the masked function.
        ab = fn(a, b, *args, **kwargs)

        # If the fade wouldn't change the blended image, don't waste
        # time trying to calculate the effect.
        if fade == 1.0:
            return ab

        # Apply the fade and return the result.
        ab = a + (ab - a) * fade
        return ab
    return wrapper


def can_mask(fn: Blend) -> Blend:
    """Apply a blending mask to the image."""
    @wraps(fn)
    def wrapper(
        a: ImgAry,
        b: ImgAry,
        mask: Union[None, ImgAry] = None,
        *args, **kwargs
    ) -> ImgAry:
        # Get the blended image from the decorated function.
        ab = fn(a, b, *args, **kwargs)

        # If there wasn't a mask passed in, don't waste time
        # trying to mask the effects.
        if mask is None:
            return ab

        # Apply the mask and return the result.
        ab = a * (1 - mask) + ab * mask
        return ab
    return wrapper


def will_clip(fn: Blend) -> Blend:
    """Blends that use division or unbounded addition or
    subtraction can overflow the scale of the image. This will
    keep the image in scale by clipping the values below zero
    to zero and the values above one to one.
    """
    @wraps(fn)
    def wrapper(a: ImgAry, b: ImgAry, *args, **kwargs) -> ImgAry:
        ab = fn(a, b, *args, **kwargs)
        ab[ab < 0.0] = 0.0
        ab[ab > 1.0] = 1.0
        return ab
    return wrapper


def will_colorize(fn: Blend) -> Blend:
    """Ensure the images have the same number of color
    channels.
    """
    @wraps(fn)
    def wrapper(
        a: ImgAry,
        b: ImgAry,
        colorize: bool = True,
        *args, **kwargs
    ) -> ImgAry:
        # If the image have different numbers of color channels,
        # add color channels to the one with the fewest.
        if colorize:
            a_dims = len(a.shape)
            b_dims = len(b.shape)
            if a_dims + 1 == b_dims and b.shape[-1] == 3:
                a = grayscale_to_rgb(a)
            elif b_dims + 1 == a_dims and a.shape[-1] == 3:
                b = grayscale_to_rgb(b)

        # Blend and return.
        ab = fn(a, b, *args, **kwargs)
        return ab
    return wrapper


def will_match_size(fn: Blend) -> Blend:
    """If the given images are different sizes, increase the size of
    the smaller image to match the larger image. Since this affects
    the size of the images, this will need to go before any decorators
    that use the original images to affect the resulting image.
    """
    @wraps(fn)
    def wrapper(a: ImgAry, b: ImgAry, *args, **kwargs) -> ImgAry:
        # Calculate the new size of the images.
        size = tuple(max(dim) for dim in zip(a.shape, b.shape))

        # Resize the dimensions of the arrays that are smaller than
        # the new array size.
        a = pad_array(a, size)
        b = pad_array(b, size)

        # Blend and return.
        ab = fn(a, b, *args, **kwargs)
        return ab
    return wrapper

