"""
model
=====

Types for :mod:`pjimg.filters`
"""
from typing import Callable, NewType

from pjimg.util import ImgAry


# Typing.
Color = NewType('Color', tuple[str, str])
ColorDict = NewType('ColorDict', dict[str, Color])
Filter = Callable[..., ImgAry]

# Registry of filter functions.
filters: dict[str, Filter] = dict()
