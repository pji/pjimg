"""
util
====

Utility functions for :mod:`pjimg.filters`.
"""
from pjimg.filters.constants import COLORS
from pjimg.filters.model import Color


# Functions.
def get_color_for_key(colorkey: str) -> Color:
    return COLORS[colorkey]
