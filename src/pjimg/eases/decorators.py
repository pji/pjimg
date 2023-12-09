"""
Decorators
==========

Decorators for :mod:`pjimg.eases`.

.. autofunction:: pjimg.eases.register
.. autofunction:: pjimg.eases.will_scale
"""
from functools import wraps
from typing import Callable

import numpy as np

from pjimg.eases.model import Ease
from pjimg.util import NumAry


# Decorators.
def register(registry: dict[str, Ease]) -> Callable[[Ease,], Ease]:
    """Registers the decorated function under the function's name
    in the given registry dictionary.
    
    :param registry: The registry to register the given function in.
    :return: The registration :mod:`function` pointed to the given
        registry.
    :rtype: function
    """
    def decorator(fn: Ease) -> Ease:
        key = fn.__name__
        registry[key] = fn
        return fn
    return decorator


def will_scale(fn: Ease) -> Ease:
    """These eases only work for values between zero and one. If
    given values outside of that range, it will scale the values
    down to that range, run the easing function, then scale the
    values back up to the original range.
    
    :param fn: The decorated easing functions.
    :return: The now wrapped :mod:`function`.
    :rtype: function
    """
    @wraps(fn)
    def wrapper(a: NumAry, *args, **kwargs) -> NumAry:
        # Make a defensive copy before acting on the array.
        a = a.copy()

        # Only scale data that isn't within zero to one.
        scale = 1.0
        scale_offset = 0.0
        if np.min(a) < 0.0 or np.max(a) > 1.0:
            scale_offset = np.min(a)
            a -= scale_offset
            scale = np.max(a)
            a /= scale

        # Perform the ease.
        a = fn(a)

        # If the data was scaled, undo the scaling.
        if scale != 1.0 or scale_offset != 0.0:
            a *= scale
            a += scale_offset

        return a
    return wrapper
