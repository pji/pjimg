"""
Decorators
==========

These decorators provide useful functions for building new filters:

.. autofunction:: pjimg.filters.register
.. autofunction:: pjimg.filters.processes_by_grayscale_frame
.. autofunction:: pjimg.filters.uses_uint8
.. autofunction:: pjimg.filters.will_square

"""
from functools import wraps
from typing import Callable

import numpy as np

from pjimg.filters.model import Filter
from pjimg.util import ImgAry, X, X_, Y, Y_, Z, Z_


# Names available for import.
__all__ = [
    'register', 'processes_by_grayscale_frame', 'uses_uint8', 'will_square',
]


# Docstring additions.
docstrings = {
    'process': """
    .. warning::
       This filter uses a third-party library that cannot handle 
       color or three-dimensional arrays. The filter itself will 
       be able to handle three-dimensional arrays, but the filter 
       will affect each two-dimensional slice individually.
    
    """,
    'square': """
    .. warning::
       This function works best if you provide it a square image.
       If you provide image data that doesn\'t have equal sized X
       and Y axes, it will square them itself for processing then
       trim them back to the original shape after. This may
       introduce unwanted artifacts into the image.
    
    """,
}


# Decorators.
def register(registry: dict[str, Filter]) -> Callable[[Filter,], Filter]:
    """Registers the decorated function under the function's name
    in the given registry dictionary.
    
    :param registry: The registry to register the given function in.
    :return: The registration :mod:`function` pointed to the given
        registry.
    :rtype: function
    """
    def decorator(fn: Filter) -> Filter:
        key = fn.__name__
        registry[key] = fn
        return fn
    return decorator


def processes_by_grayscale_frame(fn: Filter) -> Filter:
    """If the given array is more than two dimensions, iterate
    through each two dimensional slice. This is used when the
    filter can't handle more than two dimensions in an array.
    """
    if fn.__doc__:
        fn.__doc__ += docstrings['process']
    
    @wraps(fn)
    def wrapper(a: ImgAry, *args, **kwargs) -> ImgAry:
        if len(a.shape) > 2:
            frames = [fn(frame, *args, **kwargs) for frame in a]
            out = np.array(frames)
        else:
            out = fn(a, *args, **kwargs)
        return out
    return wrapper


def uses_uint8(fn: Filter) -> Filter:
    """Converts the image data from floats to ints."""
    @wraps(fn)
    def wrapper(a: ImgAry, *args, **kwargs) -> ImgAry:
        # The wrapped function requires the image data be 8-bit
        # unsigned integers. If it's not, do the conversion.
        original_type = a.dtype
        if original_type != np.uint8:
            a = (a * 0xff).astype(np.uint8)

        # Pass the converted array to the wrapped function.
        a = fn(a, *args, **kwargs)

        # Ensure the image data is back to the type that was
        # originally passed to the function when it is returned.
        if original_type != a.dtype:
            a = a.astype(original_type) / 0xff
        return a
    return wrapper


def will_square(fn: Filter) -> Filter:
    """The array needs to have equal sized X and Y axes. The result
    will be sliced to the size of the original array.
    """
    if fn.__doc__:
        fn.__doc__ += docstrings['square']
    
    @wraps(fn)
    def wrapper(a: ImgAry, *args, **kwargs) -> ImgAry:
        # Determine if the Y and X axes aren't square.
        old_size = None
        if a.shape[X_] != a.shape[Y_]:
            old_size = a.shape
            largest = max(a.shape[Y_:])
            new_size = (*a.shape[:Y_], largest, largest)
            new_a = np.zeros(new_size, dtype=a.dtype)
            x_start = (largest - old_size[X_]) // 2
            x_end = x_start + old_size[X_]
            y_start = (largest - old_size[Y_]) // 2
            y_end = y_start + old_size[Y_]
            new_a[..., y_start:y_end, x_start:x_end] = a
            a = new_a
            del new_a

        # Send to the wrapped function.
        a = fn(a, *args, **kwargs)

        # Resize result back to the size of the original image if
        # needed before returning.
        if old_size:
            y_start = (a.shape[Y_] - old_size[Y_]) // 2
            y_end = y_start + old_size[Y]
            x_start = (a.shape[X_] - old_size[X_]) // 2
            x_end = x_start + old_size[X_]
            a = a[..., y_start:y_end, x_start:x_end]
        return a
    return wrapper
