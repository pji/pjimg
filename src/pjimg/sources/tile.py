"""
tile
~~~~

Sources that tile shapes over the image.

.. autoclass:: pjimg.sources.Tile

"""
from typing import Any, Optional, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from pjimg.sources.constants import DOWN, LEFT, RIGHT, UP
from pjimg.sources.decorators import register
from pjimg.sources.model import Seed, TilePattern
from pjimg.sources.noise import Noise
from pjimg.util import ImgAry, IntAry, Loc, Size, X, Y, Z
from pjimg.util.util import translate_by_polar_coords


# Two-dimensional axes.
y, x = 0, 1

# Registry of tile patterns.
tile_patterns: dict[str, TilePattern] = dict()


# Pattern classes.
@register(tile_patterns)
class Hexagon(TilePattern):
    """Tile with hexagons."""
    sides: int = 6
    mod_next_o: float = 2
    
    def __init__(
        self, size: Size,
        vp: float,
        gap: float,
        rotation: float,
        loc: Loc = (0, 0, 0)
    ) -> None:
        super().__init__(size, vp, gap, rotation, loc)
    
    @property
    def h_row(self) -> float:
        h = 2 * self.vp
        h_side = 2 * (self.vp * np.sin(np.pi / self.sides))
        return h + 3 * self.vgap / 2 + h_side        
    
    @property
    def x_start(self) -> float:
        w_col = 2 * self.sp + self.gap
        cols = self.home[X] // w_col + 2
        return self.home[X] - w_col * cols
    

@register(tile_patterns)
class Octagon(TilePattern):
    """Tile with octagons."""
    sides: int = 8
    mod_next_o: float = 2
    
    def __init__(
        self, size: Size,
        vp: float,
        gap: float,
        rotation: float,
        loc: Loc = (0, 0, 0)
    ) -> None:
        super().__init__(size, vp, gap, rotation, loc)

    @property
    def h_row(self) -> float:
        h_side = 2 * (self.vp * np.sin(np.pi / self.sides))
        return 2 * self.sp + h_side + 3 * self.gap / 2
    
    @property
    def next_p(self) -> float:
        try:
            return self._next_p
        except AttributeError:
            self._next_p = 2 * self.sp + self.gap
            return self._next_p
    
    @property
    def x_start(self) -> float:
        w_col = (2 * self.next_p) * np.cos(2 * self.vso)
        cols = self.home[X] // w_col + 2
        start = self.home[X] - w_col * cols
        return start

    @property
    def y_start(self) -> float:
        return self.home[Y] - self.h_row * self.rows

    # Public methods.            
    def get_vertices(
        self, center: tuple[float, float],
        o: float,
        vp: Optional[float] = None,
        sides: Optional[int] = None,
        vso: Optional[float] = None
    ) -> NDArray[np.int32]:
        o -= self.vso
        return super().get_vertices(center, o, vp, sides, vso)
    

@register(tile_patterns)
class OctagonWithSquares(TilePattern):
    """Tile with octagons."""
    sides: int = 8
    mod_next_o: float = 2
    
    def __init__(
        self, size: Size,
        vp: float,
        gap: float,
        rotation: float,
        loc: Loc = (0, 0, 0)
    ) -> None:
        super().__init__(size, vp, gap, rotation, loc)

    @property
    def h_row(self) -> float:
        return 2 * self.sp + self.h_side + 3 * self.gap / 2
    
    @property
    def h_side(self) -> float:
        return 2 * (self.vp * np.sin(np.pi / self.sides))
    
    @property
    def next_p(self) -> float:
        try:
            return self._next_p
        except AttributeError:
            self._next_p = 2 * self.sp + self.gap
            return self._next_p
    
    @property
    def x_start(self) -> float:
        w_col = (2 * self.next_p) * np.cos(2 * self.vso)
        cols = self.home[X] // w_col + 2
        start = self.home[X] - w_col * cols
        return start

    @property
    def y_start(self) -> float:
        return self.home[Y] - self.h_row * self.rows

    # Public methods.            
    def get_vertices(
        self, center: tuple[float, float],
        o: float,
        vp: Optional[float] = None,
        sides: Optional[int] = None,
        vso: Optional[float] = None
    ) -> list[NDArray[np.int32]]:
        o -= self.vso
        oct_center = center[:]
        octagon = super().get_vertices(oct_center, o, vp, sides, vso)
        
        o += self.vso
        s1_center_o = 0
        s1_center_x = self.get_next_center(oct_center, UP)[0][x]
        s1_center_p = abs(oct_center[x] - s1_center_x)
        s1_center = translate_by_polar_coords(
            oct_center, s1_center_p, s1_center_o
        )
        s1_sp = s1_center_p - self.sp - self.gap
        s1_p = s1_sp / np.cos(np.pi / 4)
        s1_o = o + np.pi / 4
        s1_vso = np.pi / 4
        s1 = super().get_vertices(s1_center, s1_o, s1_p, 4, s1_vso)
        
        return [octagon, s1]
    

@register(tile_patterns)
class Square(TilePattern):
    """Tile with squares."""
    sides: int = 4
    
    def __init__(
        self, size: Size,
        vp: float,
        gap: float,
        rotation: float,
        loc: Loc = (0, 0, 0)
    ) -> None:
        super().__init__(size, vp, gap, rotation, loc)


@register(tile_patterns)
class Triangle(TilePattern):
    """Tile with triangles."""
    sides: int = 3
    mod_next_o: float = 1 / 2
    
    def __init__(
        self, size: Size,
        vp: float,
        gap: float,
        rotation: float,
        loc: Loc = (0, 0, 0)
    ) -> None:
        super().__init__(size, vp, gap, rotation, loc)
    
    @property
    def cols(self) -> int:
        return int(self.home[X] // self.w_col + 2)
    
    @property
    def orient_start(self) -> float:
        if self.rows % 2 == self.cols % 2:
            return UP
        return DOWN

    @property
    def h_row(self) -> float:
        h = self.vp + self.sp
        return h + self.gap / 2 + self.vgap / 2

    @property
    def w_col(self) -> float:
        return self.next_p * np.cos(np.pi / (self.sides * 2))
    
    @property
    def x_start(self) -> float:
        return self.home[X] - self.w_col * self.cols
    
    @property
    def y_start(self) -> float:
        orient = self.orient_start
        y_mod = 1 if orient == UP else -1
        h = self.vp + self.sp
        y_home = self.home[Y] + y_mod * (self.vp - h / 2 + self.gap / 4)
        return y_home - self.h_row * self.rows
    
    # Public methods.
    def get_next_row_start(
        self, last_row: tuple[float, float],
        last_orient: float
    ) -> tuple[tuple[float, float], float]:
        row_p = self.next_p if last_orient == UP else 2 * self.vp + self.vgap
        orient = UP if last_orient == DOWN else DOWN
        return ((last_row[y] + row_p, last_row[x]), orient)


# Source classes.
class Tile(Noise):
    """Tile a space with polygons.
    
    :param pattern: The tiling pattern to use when tiling the space.
        Valid values are available as the keys of the `sources.tile_patterns`
        registry.
    :param radius: The size of an individual tile. What this measures
        will depend on the specific tile pattern, however in simple
        patterns it should be the distance from the center of a single
        tile to the vertices of the tile in pixels. The name is a
        reference to the distance being the radius of the circle
        enclosing the tile.
    :param gap: The distance between two neighboring tiles. What this
        measures will depend on the specific tile pattern, however in
        simple patterns it should be the distance between from the side
        of one tile to the closest point on the neighboring tile.
    :param rotation: (Optional.) The angle to rotate each individual
        tile in radians. (Future versions may change this to degrees for
        consistency.) Defaults to `0`.
    :param color: (Optional.) The color of each tile. Defaults to `1.0`.
    :param drop: (Optional.) The likelihood a tile is not placed. This
        is a percentage chance within the range 0.0 <= x <= 1.0.
        Defaults to `0.0`.
    :param seed: (Optional.) A seed value for the random number generator
        used to determine whether tiles are dropped. Defaults to the
        generator running without a seed.
    :param seed_img: (Optional.) Image data that is used to determine the
        likelihood a tile is dropped from the pattern. The drop percentage
        is based on the average of the values within the same area the
        tile would be in on the final image. Defaults to `None`.
    :return: :class:Tile object.
    :rtype: sources.tile.Tile
    
    Usage::
    
        >>> # Create a tiled pattern in a 1280x720 image.
        >>> size = (1, 720, 1280)
        >>> pattern = 'triangle'
        >>> radius = 20
        >>> gap = 3
        >>> source = Tile(pattern=pattern, radius=radius, gap=gap)
        >>> img = source.fill(size)

    .. figure:: images/tile.jpg
       :alt: Tile pattern in a 1280x720 image.
       
       The image data created by the usage example.
    """
    def __init__(
        self, pattern: str,
        radius: int,
        gap: int,
        rotation: float = 0.0,
        color: float = 1.0,
        drop: float = 0.0,
        seed: Seed = None,
        seed_img: Optional[ImgAry] = None
    ) -> None:
        self.pattern = pattern
        self.radius = radius
        self.gap = gap
        self.rotation = rotation
        self.color = color
        self.drop = drop
        self.seed_img = seed_img
        super().__init__(seed)
    
    def fill(self, size: Size, loc: Loc = (0, 0, 0)) -> ImgAry:
        pattern_type = tile_patterns[self.pattern]
        pattern = pattern_type(
            size, self.radius, self.gap, self.rotation, loc
        )
        color = int(self.color * 0xff)
        line = cv2.LINE_AA
        
        # Configure the tiling.
        row_start = (pattern.y_start, pattern.x_start)
        row_orient = pattern.orient_start
        center = row_start
        orient = row_orient

        # Tile the polygons.
        a = np.zeros(size[1:], dtype=np.uint8)
        while center[y] < size[Y] + pattern.h_row:
            while center[x] < size[X] + pattern.vp:

                # Make polygons.
                modo = orient + pattern.rotation
                vertices = pattern.get_vertices(center, modo)
                if isinstance(vertices, list):
                    for polygon in vertices:
                        self._draw_polygon(a, polygon, color, line)
                else:
                    self._draw_polygon(a, vertices, color, line)
        
                # Find next polygon.
                center, orient = pattern.get_next_center(center, orient)
            
            # Find the next row.
            row_start, row_orient = pattern.get_next_row_start(
                row_start, row_orient
            )
            orient = row_orient
            center = row_start

        # Add back the Z axis and return.
        a = a[np.newaxis, :, :]
        a = np.tile(a, (size[Z], 1, 1))
        return a.astype(float) / 255
    
    # Private methods.
    def _draw_polygon(
        self, a: IntAry,
        vertices: NDArray[np.int32],
        color: int,
        line: int
    ) -> None:
        drop = self.drop
        if self.seed_img is not None:
            drop = 1 - average_color_in_shape(self.seed_img, vertices)
        if self._rng.random([1,])[0] > drop:
            cv2.fillConvexPoly(a, vertices, color=color, lineType=line)


# Utility functions.
def average_color_in_shape(
    a: ImgAry,
    vertices: NDArray[np.int32]
) -> float:
    a = np.squeeze(a)
    mask = np.zeros(a.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, vertices, color=0xff)
    masked = a[mask == 0xff]
    
    if not np.isnan(masked).all():
        return np.nanmean(masked)
    return 0.0


if __name__ == '__main__':
    from pjimg.util.debug import print_array
    
    size = (1, 8, 8)
    pattern = 'triangle'
    radius = 2
    gap = 1
    
    tile = Tile(pattern, radius, gap)
    a = tile.fill(size)
    
    a *= 0xff
    a = a.astype(np.uint8)
    print_array(a, depth=2)
