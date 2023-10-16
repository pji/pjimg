"""
Noises
------

Sources that generate psuedorandom noise.

.. autoclass:: pjimg.sources.Noise
.. autoclass:: pjimg.sources.Embers

"""
from typing import Sequence, Union

import cv2
import numpy as np
from numpy.random import default_rng

from pjimg.sources.model import Seed, Source
from pjimg.util import ImgAry, Loc, Size, X, Y, Z


# Public classes.
class Noise(Source):
    """Create continuous-uniformly distributed random noise with a
    seed value to allow the noise to be regenerated in a predictable
    way.

    :param seed: (Optional.) An int, bytes, or string used to seed
        therandom number generator used to generate the image data.
        If no value is passed, the RNG will not be seeded, so
        serialized versions of this source will not product the
        same values. Note: strings that are passed to seed will
        be converted to UTF-8 bytes before being converted to
        integers for seeding.
    :return: :class:Noise object.
    :rtype: sources.noise.Noise

    Usage::
    
        >>> # Create static to fill a 1280x720 image.
        >>> size = (1, 720, 1280)
        >>> source = Noise(seed='spam')
        >>> img = source.fill(size)

    .. figure:: images/noise.jpg
       :alt: Static filling a 1280x720 image.
       
       The image data created by the usage example.
    
    """
    def __init__(self, seed: Seed = None) -> None:
        """Initialize an instance of Noise."""
        # Store the seed for potential serialization.
        self.seed = seed

        # This seeds the random number generator. The code here is
        # maybe a bit opaque. Think about changing it in the future.
        self._rng = self._get_rng(seed)

    # Properties.
    def _get_rng(self, seed: Seed) -> np.random._generator.Generator:
        # The seed value for numpy.default_rng cannot be a string.
        # You can't convert directly from string to integer, so
        # convert the string to bytes.
        if isinstance(seed, str):
            seed = bytes(seed, 'utf_8')

        # The seed value for numpy.default_rng needs to be an integer.
        if isinstance(seed, bytes):
            seed = int.from_bytes(seed, 'little')
        return default_rng(seed)

    # Public methods.
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
        # Random number generation is linear and unidirectional. In
        # order to give the illusion of their being a space to move
        # in, we define the location of the first number generated
        # as the origin of the space (so: [0, 0, 0]). We then will
        # make the negative locations in the space the reflection of
        # the positive spaces.
        new_loc = [abs(n) for n in loc]

        # To simulate positioning within a space, we need to burn
        # random numbers from the generator. This would be easy if
        # we were just generating single dimensional noise. Then
        # we'd only need to burn the first numbers from the generator.
        # Instead, we need to burn until with get to the first row,
        # then accept. Then we need to burn again until we get to
        # the second row, and so on. This implementation isn't very
        # memory efficient, but it should do the trick.
        new_size = [s + l for s, l in zip(size, new_loc)]
        a = self._rng.random(new_size)
        slices = tuple(slice(n, None) for n in new_loc)
        a = a[slices]
        return a


class Embers(Noise):
    """Fill a space with bright points or dots that resemble embers
    or stars.

    :param depth: (Optional.) The number of different sizes of dots
        to create.
    :param threshold: (Optional.) Embers starts by generating random
        values for each point. This sets the minimum value to keep in
        the output. It's a percentage, and the lower the value the
        more points are kept.
    :param seed: (Optional.) An int, bytes, or string used to seed
        therandom number generator used to generate the image data.
        If no value is passed, the RNG will not be seeded, so
        serialized versions of this source will not product the
        same values. Note: strings that are passed to seed will
        be converted to UTF-8 bytes before being converted to
        integers for seeding.
    :param ease: (Optional.) The easing function to use on the
        generated noise.
    :return: :class:`Embers` object.
    :rtype: sources.noise.Embers
    
    Usage::
    
        >>> # Create embers in a 1280x720 image.
        >>> size = (1, 720, 1280)
        >>> source = Embers(depth=6, seed='spam')
        >>> img = source.fill(size)

    .. figure:: images/Embers.jpg
       :alt: Embers in a 1280x720 image.
       
       The image data created by the usage example.
    
    """
    def __init__(
        self, depth: int = 1,
        threshhold: float = 0.9998,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.depth = depth
        self.threshhold = threshhold

    # Public methods.
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
        mag = 1.0
        out = np.zeros(size, dtype=float)
        for layer in range(self.depth):
            # Use the magnification to determine the size of the noise
            # to get.
            fill_size = [size[0], *(int(n // mag) for n in size[1:])]

            # Get the noise to work with.
            a = super().fill(fill_size, loc)

            # Use the threshold to turn it into a sparse collection
            # of points. Then scale to increase the apparent difference
            # in brightness.
            a = a - self.threshhold
            a[a < 0] = 0.0
            a[a > 0] = a[a > 0] * 0.25
            a[a > 0] = a[a > 0] + 0.75

            # Resize to increase the size of the points.
            resized = np.zeros(size, dtype=a.dtype)
            for i in range(resized.shape[Z]):
                frame = np.zeros(a.shape[Y:3], dtype=a.dtype)
                frame = a[i]
                resized[i] = cv2.resize(frame, (size[X], size[Y]))

            # Blend the layer with previous layers.
            out = self._blend(out, resized)

            mag = mag * 1.5

        return out

    # Private methods.
    def _blend(self, a: ImgAry, b: ImgAry) -> ImgAry:
        ab = a.copy()
        ab[b > a] = b[b > a]
        return ab
