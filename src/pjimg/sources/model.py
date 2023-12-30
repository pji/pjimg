"""
model
-----

Types used for :mod:`pjimg.sources`.
"""
from abc import ABC, abstractmethod
from inspect import signature
from typing import Any, Optional, Union

import numpy as np
from numpy.typing import NDArray

from pjimg.sources.constants import DOWN, LEFT, RIGHT, UP
from pjimg.util import *


# Two-dimensional axes.
y, x = 0, 1

# Typing.
Seed = Union[None, int, str, bytes]


# Base classes.
class Serializable(ABC):
    """An object that can be serialized as either a :class:`tuple` or
    a :class:`dict`.
    """
    def __eq__(self, other):
        """Determine the equality of this and another object."""
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.asdict() == other.asdict()

    def __repr__(self):
        """Return a string representation of the object."""
        cls = self.__class__.__name__
        attrs = self.asdict()
        for attr in attrs:
            if isinstance(attrs[attr], str):
                attrs[attr] = f"'{attrs[attr]}'"
        args = [f'{k}={attrs[k]}' for k in attrs]
        args_str = ", ".join(args)
        if len(args_str) > 30:
            args_str = args_str[:10] + '...' + args_str[-10:]
        return f'{cls}({args_str})'

    def asargs(self) -> tuple[Any, ...]:
        """Serialize the object to a tuple."""
        sig = signature(self.__init__)                      # type: ignore
        params = sig.parameters
        return tuple(getattr(self, p) for p in params)

    def asdict(self) -> dict[str, Any]:
        """Serialize the object to a dictionary."""
        sig = signature(self.__init__)                      # type: ignore
        params = sig.parameters
        return {k: getattr(self, k) for k in params}


class Source(Serializable):
    """A source of image data."""
    @abstractmethod
    def fill(
        self, size: Size,
        loc: Loc = (0, 0, 0)
    ) -> ImgAry:
        """Fill a volume with image data.

        :param size: The size of the volume of image data to generate.
        :param loc: (Optional.) How much to shift the starting point
            for the noise generation along each axis.
        :return: An :class:`numpy.ndarray` with image data.
        :rtype: numpy.ndarray
        """


class TilePattern(ABC):
    """Base class for tiling patterns.
    
    :param size: The size of the image data to be tiled.
    :param vp: The distance in pixels from the center of a tile to each
        of its vertices.
    :param gap: The distance in pixel between a side of a tile to the
        nearest point of its neighbor.
    :param rotation: The amount in radians to rotate each tile.
    :param loc: (Optional.) How much to offset the position of the tiles
        in each dimension of the image data.
    :return: A :class:`TilePattern` object.
    :rtype: pjimg.sources.model.TilePattern
    """
    sides: int
    mod_next_o: float = 1

    def __init__(
        self, size: Size,
        vp: float,
        gap: float,
        rotation: float,
        loc: Loc = (0, 0, 0)
    ) -> None:
        self.loc = loc
        self.size = size
        self.vp = vp
        self.gap = gap
        self.rotation = rotation
        
        self.vso = np.pi / self.sides
        self.sp = np.cos(self.vso) * self.vp
        self.vgap = gap / np.cos(self.vso)
    
    @property
    def h_row(self) -> float:
        return 2 * self.vp + self.vgap
    
    @property
    def home(self) -> Loc:
        return self._home
    
    @property
    def next_p(self) -> float:
        try:
            return self._next_p
        except AttributeError:
            self._next_p: float = 2 * self.sp + self.gap
            return self._next_p
    
    @property
    def orient_start(self) -> float:
        return DOWN
    
    @property
    def rows(self) -> int:
        return int(self.home[Y] // self.h_row + 2)
    
    @property
    def size(self) -> Size:
        return self._size
    
    @size.setter
    def size(self, size: Size) -> None:
        self._size = size
        self._home = find_center(self.size, self.loc)
    
    @property
    def x_start(self) -> float:
        w_col = self.h_row
        cols = self.home[X] // w_col + 2
        return self.home[X] - w_col * cols

    @property
    def y_start(self) -> float:
        return self.home[Y] - self.h_row * self.rows

    # Public methods.
    def get_next_center(
        self, center: tuple[float, float], orient: float
    ) -> tuple[tuple[float, float], float]:
        """Get the linear coordinates of the next tile.
        
        :param center: The linear coordinates of the current tile.
        :param orient: The orientation in the pattern of the current tile.
        :return: A :class:`tuple` containing the linear coordinates of
            the next tile in the pattern and the orientation of that tile
            in the pattern.
        :rtype: tuple
        """
        if orient == UP:
            o = -self.mod_next_o * self.vso
            orient = DOWN
        else:
            o = self.mod_next_o * self.vso
            orient = UP
        center = translate_by_polar_coords(center, self.next_p, o)
        return center, orient
            
    def get_next_row_start(
        self, last_row: tuple[float, float],
        last_orient: float
    ) -> tuple[tuple[float, float], float]:
        """Get the linear coordinates of the first tile of the next row.
        
        :param center: The linear coordinates of the first tile of the
            current row.
        :param orient: The orientation in the pattern of the first tile
            of the current row.
        :return: A :class:`tuple` containing the linear coordinates of
            the first tile in the next row of the pattern and the
            orientation of that tile in the pattern.
        :rtype: tuple
        """
        return ((
            last_row[y] + self.h_row,
            last_row[x]
        ), last_orient)
        
    def get_vertices(
        self, center: tuple[float, float],
        o: float,
        vp: Optional[float] = None,
        sides: Optional[int] = None,
        vso: Optional[float] = None
    ) -> list[NDArray[np.int32]]:
        """Get the vertices of the tile.
        
        :param center: The linear coordinates of the tile.
        :param o: The theta polar coordinate of the first vertex.
        :param vp: (Optional.) The rho polar coordinate of vertices.
            Defaults to a value based on the values given when the
            :class:`TilePattern` was initialized.
        :param sides: (Optional.) The number of sides for the tile.
            Defaults to the number of sides given to the
            :class:`TilePattern`.
        :param vso: (Optional.) The theta polar coordinate for the
            angle between a line from the center to a vertex of the
            tile and a line from the center to the center of the
            nearest side to that vertex. Defaults to a value based
            on the values given when the :class:`TilePattern` was
            initialized.
        :return: A :class:`numpy.ndarray` of the linear coordinates
            of the vertices of the tile.
        :rtype: numpy.ndarray
        """
        if vp is None:
            vp = self.vp
        if sides is None:
            sides = self.sides
        if vso is None:
            vso = self.vso
        
        vvo = 2 * vso
        return [np.array([[
            (
                vp * np.cos(o + i * vvo) + center[x],
                vp * np.sin(o + i * vvo) + center[y],
            )
            for i in range(sides)
        ]], dtype=np.int32),]
