"""
model
=====

Types for :mod:`pjimg.eases`.
"""
from typing import Callable

from pjimg.util import ImgAry


# Typing.
Ease = Callable[[ImgAry], ImgAry]


# Registry of ease functions.
eases: dict[str, Ease] = dict()
