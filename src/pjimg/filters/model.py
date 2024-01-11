"""
model
=====

Types for :mod:`pjimg.filters`
"""
from typing import Callable, NewType


# Typing.
Color = NewType('Color', tuple[str, str])
ColorDict = NewType('ColorDict', dict[str, Color])
Filter = Callable

# Registry of filter functions.
filters: dict[str, Filter] = dict()
