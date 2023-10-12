"""
model
=====

Types used in :mod:`pjimg.imgblend`.
"""
from typing import Callable

from pjimg.util.model import ImgAry

# Typing.
Blend = Callable[[ImgAry, ImgAry], ImgAry]
