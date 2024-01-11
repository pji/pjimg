"""
util
====

General utility functions for :mod:`pjimg`.

.. autofunction:: pjimg.util.find_center
.. autofunction:: pjimg.util.get_free_rotation_size_2d
.. autofunction:: pjimg.util.translate_by_polar_coords

"""
import math
from inspect import getmembers, isfunction
from typing import Callable

import numpy as np

from pjimg.util.constants import X, Y, Z
from pjimg.util.model import Size, Loc


# Exported names.
__all__ = [
    'find_center', 'get_free_rotation_size_2d', 'translate_by_polar_coords'
]


# Calculations.
def find_center(size: Size, loc: Loc = (0, 0, 0)) -> Size:
    """Find the center pixel of an image of the given size.
    
    :param size: The shape of the image data.
    :param loc: (Optional.) How much to offset the center in each
        dimension. Defaults to no offset.
    :return: A :class:`tuple` with the center location of the image.
    :rtype: tuple
    """
    return tuple([n // 2 + o for n, o in zip(size, loc)])


def get_free_rotation_size_2d(
    size: Size,
    pivot_offset: Loc = (0, 0, 0)
) -> Size:
    """Given the size of a final image, return the size of the image
    data you need to create in order to rotate the final image within
    the image data freely around the Z axis without the corners of
    the final image going outside of the bounds of the image data.
    
    Basically, it says how big the image data needs to be so the
    corners of the final image won't be black if you rotate the image.
    
    :param size: The size of the final image.
    :return: A :class:`tuple` containing the shape of the needed image
        data.
    :rtype: tuple
    """
    _, h, w = [n / 2 + abs(o) for n, o in zip(size, pivot_offset)]
    o = math.atan(h / w)
    d = 2 * int(h // math.sin(o)) + 1
    return (size[Z], d, d)


def translate_by_polar_coords(
    start: tuple[float, float],
    p: float,
    o: float
) -> tuple[float, float]:
    """Given a two-dimensional location in linear coordinates and
    a distance in polar coordinates, return the linear coordinates
    of the location that distance away from the original location.
    
    :param start: The starting location.
    :param p: The rho (distance) of the polar coordinates.
    :param o: The theta (angle) of the polar coordinates in radians.
    :return: A :class:`tuple` with the final coordinates.
    :rtype: tuple
    """
    y, x = 0, 1
    return (
        start[y] + p * np.sin(o),
        start[x] + p * np.cos(o),
    )


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
