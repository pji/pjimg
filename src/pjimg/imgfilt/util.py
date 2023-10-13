"""
util
====

Utility functions for :mod:`pjimg.imgfilt`.
"""
from pjimg.imgfilt.constants import COLORS
from pjimg.imgfilt.model import Color


# Functions.
def get_color_for_key(colorkey: str) -> Color:
    return COLORS[colorkey]
