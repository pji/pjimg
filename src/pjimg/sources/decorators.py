"""
Decorators
==========

Decorators for :mod:`pjimg.sources`.

.. autofunction:: pjimg.sources.register

"""
from functools import wraps
from typing import Callable

from pjimg.sources.model import TilePattern


# Decorators.
def register(
    registry: dict[str, type[TilePattern]]
) -> Callable[[type[TilePattern],], type[TilePattern]]:
    """Registers the decorated function under the function's name
    in the given registry dictionary.
    
    :param registry: The registry to register the given class in.
    :return: The registration :mod:`class` pointed to the given
        registry.
    :rtype: class
    """
    def decorator(fn: type[TilePattern]) -> type[TilePattern]:
        key = fn.__name__.lower()
        registry[key] = fn
        return fn
    return decorator
