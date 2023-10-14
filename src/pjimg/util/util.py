"""
util
====

General utility functions for :mod:`pjimg`.

.. autofunction:: pjimg.util.get_prefixed_functions
"""
from inspect import getmembers, isfunction
from typing import Callable


# Functions.
def get_prefixed_functions(prefix: str, obj: object) -> dict[str, Callable]:
    """Return the functions within the given object that start with
    the prefix.
    
    :param prefix: The prefix of the functions to gather.
    :param obj: The module to gather from.
    :return: A :class:`dict` of the gathered functions.
    :rtype: dict
    """
    names = getmembers(obj, isfunction)
    p_len = len(prefix)
    fns = {name[p_len:]: fn for name, fn in names if name.startswith(prefix)}
    return dict(fns)
