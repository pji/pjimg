"""
model
=====

Types for :mod:`pjimg.imgfilt`
"""
from typing import Callable, NewType


# Typing.
Color = NewType('Color', tuple[str, str])
ColorDict = NewType('ColorDict', dict[str, Color])
Filter = Callable
