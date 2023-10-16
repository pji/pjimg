"""
model
=====

Types used in :mod:`pjimg.blends`.
"""
from typing import Callable

from pjimg.util.model import ImgAry

# Typing.
Blend = Callable[[ImgAry, ImgAry], ImgAry]
