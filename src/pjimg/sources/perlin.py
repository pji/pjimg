"""
Perlin Noise
------------

Perlin noise is a specific type of unit noise that uses procedures developed
by Ken Perlin to provide a more naturalistic look to computer generated
images.

.. autoclass:: pjimg.sources.Perlin
.. autoclass:: pjimg.sources.OctavePerlin

"""
import datetime as dt
import math
from typing import Optional, Sequence, overload

import numpy as np
import torch

from pjimg.sources import unitnoise as un
from pjimg.sources.model import Seed
from pjimg.util import ImgAry, IntAry64, Loc, RatioAry, Size, X, Y, Z


# Names available for import.
__all__ = [
    'BorktavePerlin',
    'OctavePerlin',
    'Perlin',
]


# Public class.
class Perlin(un.UnitNoise):
    """A class to generate Perlin noise.

    :param unit: The number of pixels between vertices along an
        axis on the unit grid. The vertices are the locations where
        colors for the gradient are set. This is involved in setting
        the maximum size of noise that can be generated from
        the object.
    :param min: (Optional.) The minimum value of a vertex of the unit
        grid. This is involved in setting the maximum size of noise
        that can be generated from the object. Unless you have a very
        good reason, this is probably best left at the default.
    :param max: (Optional.) The maximum value of a vertex of the unit
        grid. This is involved in setting the maximum size of noise
        that can be generated from the object. Unless you have a very
        good reason, this is probably best left at the default.
    :param repeats: (Optional.) The number of times each value can
        appear on the unit grid. This is involved in setting the
        maximum size of noise that can be generated from the object.
        Unless you have a very good reason, this is probably best left
        at the default.
    :param seed: (Optional.) An int, bytes, or string used to seed
        therandom number generator used to generate the image data.
        If no value is passed, the RNG will not be seeded, so
        serialized versions of this source will not produce the
        same values. Note: strings that are passed to seed will
        be converted to UTF-8 bytes before being converted to
        integers for seeding.
    :return: :class:Perlin object.
    :rtype: sources.perlin.Perlin

    Usage::

        >>> # Create Perlin noise in a 1280x720 image.
        >>> size = (1, 720, 1280)
        >>> unit = (1, size[Y] // 5, size[Y] // 5)
        >>> source = Perlin(unit=unit, seed='spam')
        >>> img = source.fill(size)

    .. figure:: images/perlin.jpg
       :alt: Perlin noise in a 1280x720 image.

       The image data created by the usage example.

    """
    def __init__(
        self, unit: Size,
        min: int = 0x00,
        max: int = 0xff,
        repeats: int = 1,
        table: Optional[Sequence[int]] = None,
        seed: Seed = None,
        device: str = ''
    ) -> None:
        """Initialize an instance of UnitNoise."""
        super().__init__(unit, min, max, repeats, table, seed, device)

    # Public classes.
    def fill_array(self, size: Size, loc: Loc = (0, 0, 0)) -> ImgAry:
        """Fill a volume with image data.

        :param size: The size of the volume of image data to generate.
        :param loc: (Optional.) How much to shift the starting point
            for the noise generation along each axis.
        :return: An :class:`numpy.ndarray` with image data.
        :rtype: numpy.ndarray
        """
        shape = self._calc_unit_grid_shape(size)
        whole, parts = self._map_unit_grid(size, loc)
        fades = 6 * parts ** 5 - 15 * parts ** 4 + 10 * parts ** 3
        grids = self._build_grids(whole, size, shape)
        for grid in grids:
            grids[grid] = self._grad(grid, grids[grid], parts)
        a = self._interp(grids, fades)
        return (a + 1) / 2

    def fill_tensor(self, size: Size, loc: Loc = (0, 0, 0)) -> torch.Tensor:
        shape = self._calc_unit_grid_shape_tensor(size)
        whole, parts = self._map_unit_grid_tensor(size, loc)
        grids = self._build_grids_tensor(whole, size, shape)
        for grid in grids:
            grids[grid] = self._grad_tensor(grid, grids[grid], parts)
        fades = [self._fade_tensor(mesh) for mesh in parts]
        t = self._interp_tensor(grids, fades)
        t = (t + 1) / 2
        return t

    # Private methods.
    def _build_grids(
        self, whole: IntAry64,
        size: Sequence[float],
        shape: Size
    ) -> dict[str, IntAry64]:
        """Get the color for the eight vertices that surround each of
        the pixels.
        """
        grids = {}
        for key in self._hashes:
            grid_whole = whole.copy()
            for axis in range(self._axes):
                grid_whole[axis] += int(key[axis])

            a_grid = grid_whole[Z].astype(np.int64)
            for axis in (Y, X):
                a_grid = np.take(self.table, a_grid) + grid_whole[axis]

            grids[key] = a_grid
        return grids

    def _build_grids_tensor(
        self, whole: Sequence[torch.Tensor],
        size: Sequence[float],
        shape: Size
    ) -> dict[str, torch.Tensor]:
        """Get the color for the eight vertices that surround each of
        the pixels.
        """
        grids = {}
        for key in self._hashes:

            t_grid = torch.clone(whole[Z]).long()
            t_grid = t_grid.detach()
            t_grid += int(key[Z])

            for axis in (Y, X):
                mesh = torch.clone(whole[axis]).long()
                mesh = mesh.detach()
                mesh += int(key[axis])

                # This is very slow on MPS, likely because torch.take
                # hasn't been implemented for MPS. If this can be
                # refactored to avoid torch.take there is probably
                # a big speed up available. It's still a lot faster
                # than the numpy version, though.
                t_grid = torch.take(self.table_tensor, t_grid) + mesh

            grids[key] = t_grid
        return grids

    def _fade_tensor(self, t):
        """Perform the Perlin fade algorithm."""
        return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

    def _grad(self, loc_mask, grid, parts):
        """Calculate the dot product of the randomized gradient vector
        and the eight location vectors.

        This uses the optimization of the gradient function that was
        developed by Riven. At time of writing, that optimization can
        be found here::

            http://riven8192.blogspot.com/2010/08/
            calculate-perlinnoise-twice-as-fast.html

        :param loc_mask: The identifier for the vertex being worked on.
        :param grid: The vector for the vertex being worked on.
        :param parts: The relative distance from the vertex before the
            pixel and the pixel.
        :return: The dot products as a :class:`numpy.ndarray`.
        :rtype: numpy.ndarray
        """
        z = parts[Z].copy()
        y = parts[Y].copy()
        x = parts[X].copy()
        if loc_mask[0] == '1':
            z -= 1
        if loc_mask[1] == '1':
            y -= 1
        if loc_mask[2] == '1':
            x -= 1

        # Calculate the dot products and return the result.
        m = grid & 0xf

        u = x.copy()
        u[m >= 0x8] = y[m >= 8]
        u[m & 1 != 0] = -u[m & 1 != 0]

        v = z.copy()
        v[m < 0x4] = y[m < 0x4]
        v[m == 0xc] = x[m == 0xc]
        v[m == 0xe] = x[m == 0xe]
        v[m & 2 != 0] = -v[m & 2 != 0]

        out = np.zeros_like(x)
        out = u + v
        return out.astype(float)

    def _grad_tensor(self, loc_mask, grid, parts):
        """Calculate the dot product of the randomized gradient vector
        and the eight location vectors.

        This uses the optimization of the gradient function that was
        developed by Riven. At time of writing, that optimization can
        be found here::

            http://riven8192.blogspot.com/2010/08/
            calculate-perlinnoise-twice-as-fast.html

        :param loc_mask: The identifier for the vertex being worked on.
        :param grid: The vector for the vertex being worked on.
        :param parts: The relative distance from the vertex before the
            pixel and the pixel.
        :return: The dot products as a :class:`numpy.ndarray`.
        :rtype: numpy.ndarray
        """
        z = torch.clone(parts[Z])
        y = torch.clone(parts[Y])
        x = torch.clone(parts[X])
        if loc_mask[0] == '1':
            z -= 1
        if loc_mask[1] == '1':
            y -= 1
        if loc_mask[2] == '1':
            x -= 1

        # Calculate the dot products and return the result.
        m = grid & 0xf

        u = torch.clone(x)
        u[m >= 0x8] = y[m >= 8]
        u[m & 1 != 0] = -u[m & 1 != 0]

        v = torch.clone(z)
        v[m < 0x4] = y[m < 0x4]
        v[m == 0xc] = x[m == 0xc]
        v[m == 0xe] = x[m == 0xe]
        v[m & 2 != 0] = -v[m & 2 != 0]

        out = u + v
        return out.float()


# Octave unit noise classes.
defaults = un.OctaveNoiseDefaults(6, -4, 24, 4)
OctavePerlin = un.octave_noise_factory(Perlin, defaults)
BorktavePerlin = un.octave_noise_factory(Perlin, defaults, True)
