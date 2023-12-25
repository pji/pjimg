"""
tile
~~~~

Sources that tile shapes over the image.
"""
from typing import Any, Optional, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from pjimg.sources.model import Seed
from pjimg.sources.noise import Noise
from pjimg.util import ImgAry, Loc, Size, X, Y, Z


# Common orientation angles.
DOWN = np.pi / 2
LEFT = np.pi
RIGHT = 0
UP = 3 * np.pi / 2


# Classes.
class Tile(Noise):
    """Tile a space with polygons."""
    def __init__(
        self, pattern: str,
        rho: int,
        gap: int,
        rotation: float = 0,
        color: int = 0xff,
        drop: float = 0,
        seed: Seed = None,
        seed_img: Optional[ImgAry] = None
    ) -> None:
        self.pattern = pattern
        self.rho = rho
        self.gap = gap
        self.rotation = rotation
        self.color = color
        self.drop = drop
        self.seed_img = seed_img
        super().__init__(seed)
        
        self.patterns = {
            'hexagon': self._hexagon,
            'square': self._square,
            'triangle': self._triangle,
        }
    
    def fill(self, size: Size, loc: Loc = (0, 0, 0)) -> ImgAry:
        meth = self.patterns[self.pattern]
        home = find_center(size, loc)
        return meth(size, home, loc)
    
    def _hexagon(
        self, size: Size, home: Size, loc: Loc = (0, 0, 0)
    ) -> ImgAry:
        y, x = 0, 1
        sides = 6
        vp = self.rho
        sp = np.cos(np.pi / 6) * vp
        gap = self.gap
        rotation = self.rotation
        color = int(self.color * 0xff)
        line = cv2.LINE_AA
        
        # Get the Y center for the first tile.
        h = 2 * vp
        vgap = gap / np.cos(np.pi / 6)
        h_side = 2 * (vp * np.sin(np.pi / 6))
        h_row = h + 3 * vgap / 2 + h_side
        rows = home[Y] // h_row + 2
        y_start = home[Y] - h_row * rows
        
        # Get the X center for the first tile.
        w_col = 2 * sp + gap
        cols = home[X] // w_col + 2
        x_start = home[X] - w_col * cols
        
        # Configure the tiling.
        a = np.zeros(size[1:], dtype=np.uint8)
        orient = UP
        if not rows % 2:
            orient = DOWN
        next_p = 2 * sp + gap
        
        # Tile the hexagons.
        center = (y_start, x_start)
        row_orient = orient
        while center[y] < size[Y] + h_row:

            while center[x] < size[X] + vp:

                # Make hexagon.
                mod_theta = orient + rotation
                vertices = self._get_vertices(center, sides, vp, mod_theta)
                cv2.fillConvexPoly(a, vertices, color=color, lineType=line)
        
                # Find next square.
                if orient == UP:
                    theta = -2 * np.pi / 6
                    orient = DOWN
                else:
                    theta = 2 * np.pi / 6
                    orient = UP
                center = (
                    center[y] + ((2 * sp + gap) * np.sin(theta)),
                    center[x] + ((2 * sp + gap) * np.cos(theta)),
                )
            
            # Find the next row.
            y_start += h_row
            orient = row_orient
            center = (y_start, x_start)            

        # Add back the Z axis and return.
        a = a[np.newaxis, :, :]
        a = np.tile(a, (size[Z], 1, 1))
        return a.astype(float) / 255

    def _square(
        self, size: Size, home: Size, loc: Loc = (0, 0, 0)
    ) -> ImgAry:
        y, x = 0, 1
        sides = 4
        vp = self.rho
        sp = np.cos(np.pi / 4) * vp
        gap = self.gap
        rotation = self.rotation
        color = int(self.color * 0xff)
        line = cv2.LINE_AA
        
        # Get the Y center for the first tile.
        h = 2 * vp
        vgap = gap / np.cos(np.pi / 4)        
        h_row = h + vgap
        rows = home[Y] // h_row + 2
        y_start = home[Y] - h_row * rows
        
        # Get the X center for the first tile.
        w_col = h_row
        cols = home[X] // w_col + 2
        x_start = home[X] - w_col * cols
        
        # Configure the tiling.
        a = np.zeros(size[1:], dtype=np.uint8)
        orient = UP
        if not rows % 2:
            orient = DOWN
        next_p = 2 * sp + gap
        
        # Tile the squares.
        center = (y_start, x_start)
        row_orient = orient
        while center[y] < size[Y] + h_row:

            while center[x] < size[X] + vp:
        
                # Make square.
                mod_theta = orient + rotation
                vertices = self._get_vertices(center, sides, vp, mod_theta)
                cv2.fillConvexPoly(a, vertices, color=color, lineType=line)
            
                # Find next square.
                if orient == UP:
                    theta = -np.pi / 4
                    orient = DOWN
                else:
                    theta = np.pi / 4
                    orient = UP
                center = (
                    center[y] + ((2 * sp + gap) * np.sin(theta)),
                    center[x] + ((2 * sp + gap) * np.cos(theta)),
                )
    
            # Find the next row.
            y_start += 2 * vp + vgap
            orient = row_orient
            center = (y_start, x_start)
    
        # Add back the Z axis and return.
        a = a[np.newaxis, :, :]
        a = np.tile(a, (size[Z], 1, 1))
        return a.astype(float) / 255

    def _triangle(
        self, size: Size, home: Size, loc: Loc = (0, 0, 0)
    ) -> ImgAry:
        y, x = 0, 1
        sides = 3
        vp = self.rho
        sp = np.cos(np.pi / 3) * vp
        gap = self.gap
        rotation = self.rotation
        color = self.color
        line = cv2.LINE_AA
        
        # Get the Y center for the first tile.
        h = vp + sp
        y_home = home[Y] + (vp - h // 2)
        vgap = gap / np.cos(np.pi / 3)
        h_row = h + gap / 2 + vgap / 2
        rows = y_home // h_row + 2
        y_start = y_home - h_row * rows
        
        # Get the X center for the first tile.
        w_col = (2 * sp + gap) * np.cos(np.pi / 6)
        cols = home[X] // w_col + 2
        x_start = home[X] - w_col * cols
        
        # Configure the tiling.
        a = np.zeros(size[1:], dtype=np.uint8)
        orient = UP
        if not rows % 2:
            orient = DOWN
        next_p = 2 * sp + gap
        
        # Tile the triangles.
        center = (y_start, x_start)
        row_orient = orient
        while center[y] < size[Y] + vp:
            
            # Make a row.
            while center[x] < size[X] + vp:
                
                # Make triangle                
                mod_theta = orient + rotation
                vertices = self._get_vertices(center, sides, vp, mod_theta)
                drop = self.drop
                if self.seed_img is not None:
                    drop = 1 - average_color_in_shape(self.seed_img, vertices)
                if self._rng.random([1,])[0] > drop:
                    cv2.fillConvexPoly(a, vertices, color=color, lineType=line)
            
                # Find next triangle.
                if orient == UP:
                    theta = -np.pi / 6
                    orient = DOWN
                else:
                    theta = np.pi / 6
                    orient = UP
                center = (
                    center[y] + next_p * np.sin(theta),
                    center[x] + next_p * np.cos(theta),
                )
                
            # Find the next row.
            row_p = 2 * sp + gap if row_orient == UP else 2 * vp + vgap
            row_orient = UP if row_orient == DOWN else DOWN
            y_start += row_p
            orient = row_orient
            center = (y_start, x_start)
            
        # Add back the Z axis and return.
        a = a[np.newaxis, :, :]
        a = np.tile(a, (size[Z], 1, 1))
        return a.astype(float) / 255
        
    def _get_vertices(
        self, center: tuple[int, int],
        sides: int,
        rho: float,
        theta: float
    ) -> NDArray[np.int32]:
        angle = 2 * np.pi / sides
        return np.array([[
            (
                rho * np.cos(theta + i * angle) + center[1],
                rho * np.sin(theta + i * angle) + center[0],
            )
            for i in range(sides)
        ]], dtype=np.int32)


# Utility functions.
def find_center(size: Size, loc: Loc) -> Size:
    """Find the center pixel of an image of the given size."""
    return tuple([n // 2 + o for n, o in zip(size, loc)])


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
    
    
    # For doing stained glassing, not randomness.
#     color = 0x00
#     if not np.isnan(masked).all():
#         mean = np.nanmean(masked)
#         color = int(mean * 0xff)
#     return color
