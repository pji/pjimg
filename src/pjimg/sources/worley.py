"""
Worley Noise
------------

Worley noise scatters a series of points across the image space. Image
data is generated based on the distance from a pixel to the nearest point.

.. autoclass:: pjimg.sources.Worley
.. autoclass:: pjimg.sources.WorleyCell
.. autoclass:: pjimg.sources.OctaveWorley
.. autoclass:: pjimg.sources.OctaveWorleyCell

"""
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from pjimg.filters import gaussian_blur, unsharp_mask
from pjimg.sources.model import Seed, Source
from pjimg.sources.noise import Noise
from pjimg.util import ImgAry, IntAry64, Loc, Size, lerp


# Names available for import.
__all__ = ['OctaveWorley', 'OctaveWorleyCell', 'Worley', 'WorleyCell',]


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
        seeds = self._place_seeds(size)
        return self._map_fill(seeds, size)

    # Private methods.
    def _hypot(self, point: Loc, indices: IntAry64) -> ImgAry:
        axis_dist = [p - i for p, i in zip(point, indices)]
        return np.sqrt(sum(d ** 2 for d in axis_dist))

    def _map_fill(self, seeds: NDArray[np.int32], size: Size) -> ImgAry:
        """Map the value for each pixel in the noise."""
        indices = np.indices(size)
        max_dist = np.sqrt(sum(n ** 2 for n in size))
        dist = np.zeros(size, dtype=float)
        dist.fill(max_dist)
        for i in range(self.points):
            point = seeds[i]
            work = self._hypot(point, indices)
            dist[work < dist] = work[work < dist]

        act_max_dist = np.max(dist)
        return dist / act_max_dist
    
    def _place_seeds(self, size: Size) -> NDArray[np.int32]:
        """Place the seeds within the overall volume of noise."""
        volume_size = self.volume if self.volume else size
        volume = np.array(volume_size, dtype=float)
        seeds = self._rng.random((self.points, 3), dtype=float)
        seeds = np.around(seeds * (volume - 1))
        seeds += np.array(self.origin)
        return seeds.astype(np.int32)


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
    :param antialias: (Optional.) Whether to soften the edges of the
        cell boundaries. Essentially, if the difference between the
        distances to the two closest seeds is less than one pixel,
        the color will be a mix of the two seeds. This can cause
        unexpected results when two seeds are next to each other.
        Defaults to false.
    :return: :class:`WorleyCell` object.
    :rtype: sources.worley.WorleyCell
    
    Usage::
    
        >>> # Create Worley cell noise in a 1280x720 image.
        >>> size = (1, 720, 1280)
        >>> source = WorleyCell(points=20, seed='spam')
        >>> img = source.fill(size)

    .. figure:: images/worleycell.jpg
       :alt: WorleyCell noise in a 1280x720 image.
       
       The image data created by the usage example.
    
    """
    def __init__(
        self, points: int,
        volume: Optional[Size] = None,
        origin: Loc = (0, 0, 0),
        seed: Seed = None,
        antialias: bool = False
    ) -> None:
        self.antialias = antialias
        super().__init__(points, volume, origin, seed)
        
    
    def _map_fill(self, seeds: NDArray[np.int32], size: Size) -> ImgAry:
        """Map the value for each pixel in the noise."""
        # Assign a color for each seed.
        colors = [n / (self.points - 1) for n in range(self.points)]
        
        # Map all the distances to each point.
        indices = np.indices(size)
        max_dist = np.sqrt(sum(n ** 2 for n in size))
        dists = np.zeros((*size, self.points), dtype=float)
        
        # This line is a kludge to address a typing concern from Mypy.
        # It seems like Mypy doesn't recognize that NDArray[np.int32]
        # is multidimensional, which, fair enough, it isn't always.
        # However, it seems to be OK with me assigning the type as
        # Sequence[Loc] here despite the concern with letting the
        # NDArray go into the for loop. Should try to fix this when I
        # get more time.
        seed_list: Sequence[Loc] = [seed for seed in seeds]
        
        for i, seed in enumerate(seed_list):
            dists[:, :, :, i] = self._hypot(seed, indices)
        
        # Get the closest color and then antialias the edges.
        colormap = np.argmin(dists, -1)
        a = np.take(colors, colormap.astype(int))
        
        if self.antialias:

            # To antialias, we need the second lowest distance. To do that,
            # first we scrub the lowest distances out of the distances.
            ndists = dists.copy()
            min_indices = np.ravel(colormap)
            rows = np.prod(ndists.shape[:-1])
            row_indices = np.arange(rows) * ndists.shape[-1]
            min_row_indices = row_indices + min_indices
            raveled = np.ravel(ndists)
            raveled[min_row_indices] = max_dist
            ndists = raveled.reshape(ndists.shape)
            ncolormap = np.argmin(ndists, -1)
            b = np.take(colors, ncolormap.astype(int))
            
            # Now we need to figure out where there is a small difference
            # between the distances.
            x = np.min(ndists, -1) - np.min(dists, -1)
            m = x < 1
            
            # Interpolate the values of those edges.
            x[m] = 1 - (x[m] / 2 + .5)
            a[m] = lerp(a[m], b[m], x[m])
        
        # Return the noise.
        return a


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


class OctaveWorleyCell(Source):
    """Fill a space with octaves of Worley cell noise.

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
    :param antialias: (Optional.) Whether to soften the edges of the
        cell boundaries. Essentially, if the difference between the
        distances to the two closest seeds is less than one pixel,
        the color will be a mix of the two seeds. This can cause
        unexpected results when two seeds are next to each other.
        Defaults to false.
    :return: :class:`OctaveWorleyCell` object.
    :rtype: sources.worley.OctaveWorleyCell
    
    Usage::
    
        >>> # Create octave Worley cell noise in a 1280x720 image.
        >>> size = (1, 720, 1280)
        >>> source = OctaveWorleyCell(
        ...     octaves=3,
        ...     persistence=6,
        ...     amplitude=5,
        ...     frequency=3,
        ...     points=8,
        ...     seed='spam'
        ... )
        >>> img = source.fill(size)

    .. figure:: images/octaveworleycell.jpg
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
        seed: Seed = None,
        antialias: bool = False
    ) -> None:
        self.octaves = octaves
        self.persistence = persistence
        self.amplitude = amplitude
        self.frequency = frequency
        self.points = points
        self.volume = volume
        self.origin = origin
        self.seed = seed
        self.antialias = antialias
    
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
            octave = WorleyCell(
                points=points,
                volume=self.volume,
                origin=self.origin,
                seed=self.seed,
                antialias=self.antialias
            )
            a += octave.fill(size, loc) * amp
            max_value += amp
        a /= max_value
        return a

