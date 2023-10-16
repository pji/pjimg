"""
model
=====

Types for :mod:`pjimg.eases`.
"""
from typing import Callable

from pjimg.util import ImgAry


# Typing.
Ease = Callable[[ImgAry], ImgAry]
