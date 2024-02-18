"""
decorators
~~~~~~~~~~

General purpose decorators.
"""
from functools import wraps
from typing import Callable

from pjimg.util.model import Interpolator, NumAry


# Decorators.
def preserves_type(fn: Interpolator) -> Interpolator:
    """Ensure the datatype of the result is the same as the
    first parameter.
    """
    @wraps(fn)
    def wrapper(a: NumAry, *args, **kwargs) -> NumAry:
        a_dtype = a.dtype
        result = fn(a, *args, **kwargs)
        return result.astype(a_dtype)
    return wrapper
