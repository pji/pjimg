"""
Worley Noise
------------

Worley noise scatters a series of points across the image space. Image
data is generated based on the distance from a pixel to the nearest point.

.. autoclass:: pjimg.sources.Worley
.. autoclass:: pjimg.sources.WorleyCell
.. autoclass:: pjimg.sources.OctaveWorley

"""
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from pjimg.util import ImgAry, IntAry64, Loc, Size
from pjimg.sources.model import Seed, Source
from pjimg.sources.noise import Noise
from pjimg.util import lerp


# Names available for import.
__all__ = ['OctaveWorley', 'Worley', 'WorleyCell',]


# Public classes.
class Worley(Noise):
    """Fill a space with Worley noise.

    Worley noise is a type of cellular noise. The color value of each
    pixel within the space is determined by the distance from the pixel
    to the nearest of a set of randomly located points within the
    image. This creates structures within the noise that look like
    cells or pits.

    This implementation is heavily optimized from code found here:
    https://code.activestate.com/recipes/578459-worley-noise-generator/

    :param points: The number of cells in the image. A cell is a
        randomly placed point and the range of pixels that are
        closer to it than any other point.
    :param volume: (Optional.) The size of the volume that the points
        will be placed in. The default is for them to be evenly spread
        through the space generated during the fill.
    :param origin: (Optional.) The location of the upper-top-left
        corner of the volume that contains the points. This defaults
        to the upper-top-left corner of the space generated during the
        fill.
    :param seed: (Optional.) An int, bytes, or string used to seed
        therandom number generator used to generate the image data.
        If no value is passed, the RNG will not be seeded, so
        serialized versions of this source will not product the
        same values. Note: strings that are passed to seed will
        be converted to UTF-8 bytes before being converted to
        integers for seeding.
    :return: :class:`Worley` object.
    :rtype: sources.worley.Worley
    
    Usage::
    
        >>> # Create Worley noise in a 1280x720 image.
        >>> size = (1, 720, 1280)
        >>> source = Worley(points=20, seed='spam')
        >>> img = source.fill(size)

    .. figure:: images/worley.jpg
       :alt: Worley noise in a 1280x720 image.
       
       The image data created by the usage example.
    
    """
    def __init__(
        self, points: int,
        volume: Optional[Size] = None,
        origin: Loc = (0, 0, 0),
        seed: Seed = None
    ) -> None:
        self.points = int(points)
        self.volume = volume
        self.origin = origin
        super().__init__(seed)

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
        a = np.zeros(size, dtype=float)
        volume_size = self.volume
        if volume_size is None:
            volume_size = size
        volume = np.array(volume_size, dtype=float)

        # Place the seeds in the overall volume of noise.
        seeds = self._rng.random((self.points, 3), dtype=float)
        seeds = np.around(seeds * (volume - 1)).astype(float)
        seeds += np.array(self.origin)

        # Map the distances to the points.
        indices = np.indices(size)
        max_dist = np.sqrt(sum(n ** 2 for n in size))
        dist = np.zeros(size, dtype=float)
        dist.fill(max_dist)
        for i in range(self.points):
            point = seeds[i]
            work = self._hypot(point, indices)
            dist[work < dist] = work[work < dist]

        act_max_dist = np.max(dist)
        a = dist / act_max_dist
        return a

    # Private methods.
    def _hypot(self, point: Loc, indices: IntAry64) -> ImgAry:
        axis_dist = [p - i for p, i in zip(point, indices)]
        return np.sqrt(sum(d ** 2 for d in axis_dist))


class WorleyCell(Worley):
    """Fill a space with Worley noise that fills each cell with a
    solid color.

    Worley noise is a type of cellular noise. The color value of each
    pixel within the space is determined by the distance from the pixel
    to the nearest of a set of randomly located points within the
    image. This creates structures within the noise that look like
    cells or pits.

    This implementation is heavily optimized from code found here:
    https://code.activestate.com/recipes/578459-worley-noise-generator/

    :param points: The number of cells in the image. A cell is a
        randomly placed point and the range of pixels that are
        closer to it than any other point.
    :param volume: (Optional.) The size of the volume that the points
        will be placed in. The default is for them to be evenly spread
        through the space generated during the fill.
    :param origin: (Optional.) The location of the upper-top-left
        corner of the volume that contains the points. This defaults
        to the upper-top-left corner of the space generated during the
        fill.
    :param seed: (Optional.) An int, bytes, or string used to seed
        therandom number generator used to generate the image data.
        If no value is passed, the RNG will not be seeded, so
        serialized versions of this source will not product the
        same values. Note: strings that are passed to seed will
        be converted to UTF-8 bytes before being converted to
        integers for seeding.
    :return: :class:`WorleyCell` object.
    :rtype: sources.worley.WorleyCell
    
    Usage::
    
        >>> # Create Worley noise in a 1280x720 image.
        >>> size = (1, 720, 1280)
        >>> source = WorleyCell(points=20, seed='spam')
        >>> img = source.fill(size)

    .. figure:: images/worleycell.jpg
       :alt: WorleyCell noise in a 1280x720 image.
       
       The image data created by the usage example.
    
    """
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
        a = np.zeros(size, dtype=float)
        volume_size = self.volume
        if volume_size is None:
            volume_size = size
        volume = np.array(volume_size, dtype=float)

        # Place the seeds in the overall volume of noise.
        seeds = self._rng.random((self.points, 3), dtype=float)
        seeds = np.around(seeds * (volume - 1)).astype(float)
        seeds += np.array(self.origin)

        # Assign a color for each seed.
        colors = [n / (self.points - 1) for n in range(self.points)]

        # Map the distances to the points.
        indices = np.indices(size)
        indices = np.transpose(indices, (1, 2, 3, 0))
        dist = np.zeros((self.points, *size),
                        dtype=[('d', float), ('i', float)])
        for i in range(self.points):
            dist[i] = self._hypot(seeds[i], indices, i)
        dist = np.transpose(dist, (1, 2, 3, 0))
        dist.sort(3, order='d')
        map = dist[:, :, :, 0]['i']
        a = np.take(colors, map.astype(int))
        
        # Antialias edges.
        nmap = dist[:, :, :, 1]['i']
        m = dist[:, :, :, 1]['d'] - dist[:, :, :, 0]['d'] < 1
        b = np.take(colors, nmap.astype(int))
        x = dist[:, :, :, 1]['d'] - dist[:, :, :, 0]['d']
        x[m] = 1 - (x[m] / 2 + .5)
        a[m] = lerp(a[m], b[m], x[m])
        
        return a

    def _hypot(self, point, indices, i):
        diffs = (indices - point) ** 2
        diffs = np.sqrt(np.sum(diffs, -1))
        result = np.zeros(diffs.shape, dtype=[('d', float), ('i', float)])
        result['d'] = diffs
        result['i'].fill(i)
        return result


class OctaveWorley(Source):
    """Fill a space with octaves of Worley noise.

    Worley noise is a type of cellular noise. The color value of each
    pixel within the space is determined by the distance from the pixel
    to the nearest of a set of randomly located points within the
    image. This creates structures within the noise that look like
    cells or pits.

    This implementation is heavily optimized from code found here:
    https://code.activestate.com/recipes/578459-worley-noise-generator/

    :param octaves: The number of octaves of noise in the image. An
        octave is a layer of the noise with a different number of
        points added on top of other layers of noise.
    :param persistence: How the weight of each octave changes.
    :param amplitude: The weight of the first octave.
    :param frequency: How the number of points in each octave changes.
    :param points: The number of cells in the image. A cell is a
        randomly placed point and the range of pixels that are
        closer to it than any other point.
    :param volume: (Optional.) The size of the volume that the points
        will be placed in. The default is for them to be evenly spread
        through the space generated during the fill.
    :param origin: (Optional.) The location of the upper-top-left
        corner of the volume that contains the points. This defaults
        to the upper-top-left corner of the space generated during the
        fill.
    :param seed: (Optional.) An int, bytes, or string used to seed
        therandom number generator used to generate the image data.
        If no value is passed, the RNG will not be seeded, so
        serialized versions of this source will not product the
        same values. Note: strings that are passed to seed will
        be converted to UTF-8 bytes before being converted to
        integers for seeding.
    :return: :class:`OctaveWorley` object.
    :rtype: sources.worley.OctaveWorley
    
    Usage::
    
        >>> # Create octave Worley noise in a 1280x720 image.
        >>> size = (1, 720, 1280)
        >>> source = OctaveWorley(
        ...     octaves=3,
        ...     persistence=6,
        ...     amplitude=5,
        ...     frequency=3,
        ...     points=8,
        ...     seed='spam'
        ... )
        >>> img = source.fill(size)

    .. figure:: images/octaveworley.jpg
       :alt: Octave Worley noise in a 1280x720 image.
       
       The image data created by the usage example.
    
    """
    def __init__(
        self, octaves: int = 4,
        persistence: float = 8,
        amplitude: float = 8,
        frequency: float = 2,
        points: int = 10,
        volume: Optional[Size] = None,
        origin: Loc = (0, 0, 0),
        seed: Seed = None
    ) -> None:
        self.octaves = octaves
        self.persistence = persistence
        self.amplitude = amplitude
        self.frequency = frequency
        self.points = points
        self.volume = volume
        self.origin = origin
        self.seed = seed
    
    def fill(
        self, size: Sequence[int],
        loc: Sequence[int] = (0, 0, 0)
    ) -> ImgAry:
        a = np.zeros(tuple(size), dtype=float)
        max_value = 0.0
        for i in range(self.octaves):
            amp = self.amplitude + (self.persistence * i)
            freq = self.frequency * 2 ** i
            points = self.points * freq
            octave = Worley(
                points=points,
                volume=self.volume,
                origin=self.origin,
                seed=self.seed
            )
            a += octave.fill(size, loc) * amp
            max_value += amp
        a /= max_value
        return a
