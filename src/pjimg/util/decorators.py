"""
decorators
~~~~~~~~~~

General purpose decorators.
"""
from functools import wraps
from typing import Callable

import numpy as np

from pjimg.util.model import Interpolator, NumAry


# Decorators.
def preserves_type(fn: Interpolator) -> Interpolator:
    """Ensure the datatype of the result is the same as the
    first parameter.
    """
    @wraps(fn)
    def wrapper(a: NumAry, *args, **kwargs) -> NumAry:
        if isinstance(a, np.ndarray):
            a_dtype = a.dtype
        result = fn(a, *args, **kwargs)
        if isinstance(a, np.ndarray):
            return result.astype(a_dtype)
        return result
    return wrapper
