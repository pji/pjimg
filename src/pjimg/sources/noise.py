"""
Noises
------

Sources that generate psuedorandom noise.

.. autoclass:: pjimg.sources.Noise
.. autoclass:: pjimg.sources.Embers

"""
from typing import Sequence, Union, overload

import cv2
import numpy as np
import torch
from numpy.random import default_rng
from torchvision.transforms.v2 import Resize

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
    :param device: (Optional.) Determines the library and device
        used to generate the noise. An empty string will use
        :mod:`numpy`. Anything else will switch to :mod:`torch`
        and be used as the `device` value. Defaults to an
        empty string.
    :return: :class:Noise object.
    :rtype: sources.noise.Noise
    :usage:
        Create static to fill a 1280x720 image.

            >>> size = (1, 720, 1280)
            >>> source = Noise(seed='spam')
            >>> img = source.fill(size)

        .. figure:: images/noise.jpg
           :alt: Static filling a 1280x720 image.

           The image data created by the usage example.

    """
    def __init__(
        self, seed: Seed = None,
        *args, **kwargs
    ) -> None:
        """Initialize an instance of Noise."""
        super().__init__(*args, **kwargs)

        # Store the seed for potential serialization.
        self.seed = seed
        self._seed = self._normalize_seed(self.seed)

        # Set up the array random number generator.
        self._rng = self._get_rng()

    def _get_rng(self) -> np.random._generator.Generator:
        return default_rng(self._seed)

    def _normalize_seed(self, value: Seed) -> int | None:
        # However, RNGs need the seed to be an int. You can't convert
        # directly from string to int, so convert the string to bytes.
        if isinstance(value, str):
            value = bytes(value, 'utf_8')

        # If the passed value is bytes, convert it to an int for use
        # in seeding the RNG.
        if isinstance(value, bytes):
            value = int.from_bytes(value, 'little')

        # Return the seed normalized to int for use in seeding the RNG.
        return value

    # Public methods.
    def fill_array(self, size: Size, loc: Loc = (0, 0, 0)) -> ImgAry:
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

    def fill_tensor(self, size: Size, loc: Loc = (0, 0, 0)) -> torch.Tensor:
        """Fill a tensor with image data.

        :param size: The size of the volume of image data to generate.
        :param loc: (Optional.) How much to shift the starting point
            for the noise generation along each axis.
        :return: An :class:`torch.Tensor` with image data.
        :rtype: torch.Tensor
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
        if self._seed:
            torch.manual_seed(self._seed)
        t = torch.rand(new_size, device=self.device)
        slices = tuple(slice(n, None) for n in new_loc)
        t = t[slices]
        return t


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
    :param device: (Optional.) Determines the library and device
        used to generate the noise. An empty string will use
        :mod:`numpy`. Anything else will switch to :mod:`torch`
        and be used as the `device` value. Defaults to an
        empty string.
    :return: :class:`Embers` object.
    :rtype: sources.noise.Embers
    :usage:
        Create embers in a 1280x720 image.

            >>> size = (1, 720, 1280)
            >>> source = Embers(depth=6, seed='spam')
            >>> img = source.fill(size)

        .. figure:: images/Embers.jpg
           :alt: Embers in a 1280x720 image.

           The image data created by the usage example.

    """
    def __init__(
        self, depth: int = 1,
        threshold: float = 0.9998,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.depth = depth
        self.threshold = threshold

    # Public methods.
    def fill_array(
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
            a = super().fill_array(fill_size, loc)

            # Use the threshold to turn it into a sparse collection
            # of points. Then scale to increase the apparent difference
            # in brightness.
            a = a - self.threshold
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

    def fill_tensor(
        self, size: Size,
        loc: Loc = (0, 0, 0)
    ) -> torch.Tensor:
        """Fill a volume with image data.

        :param size: The size of the volume of image data to generate.
        :param loc: (Optional.) How much to shift the starting point
            for the noise generation along each axis.
        :return: An :class:`numpy.ndarray` with image data.
        :rtype: numpy.ndarray
        """
        mag = 1.0
        out = torch.zeros(size, dtype=torch.float32, device=self.device)
        for layer in range(self.depth):
            # Use the magnification to determine the size of the noise
            # to get.
            fill_size = [size[0], *(int(n // mag) for n in size[1:])]

            # Get the noise to work with.
            t = super().fill_tensor(fill_size, loc)

            # Use the threshold to turn it into a sparse collection
            # of points. Then scale to increase the apparent difference
            # in brightness.
            t = t - self.threshold
            t[t < 0] = 0.0
            t[t > 0] = t[t > 0] * 0.25
            t[t > 0] = t[t > 0] + 0.75

            # Resize to increase the size of the points.
            resized = torch.zeros(size, dtype=t.dtype, device=self.device)
            resizer = Resize((size[Y], size[X]))
            for i in range(resized.shape[Z]):
                frame = t[i].unsqueeze(0)
                frame = resizer(frame)
                resized[i] = frame.squeeze(0)

            # Blend the layer with previous layers.
            out = self._blend_tensor(out, resized)

            mag = mag * 1.5

        return out

    # Private methods.
    def _blend(self, a: ImgAry, b: ImgAry) -> ImgAry:
        ab = a.copy()
        ab[b > a] = b[b > a]
        return ab

    def _blend_tensor(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        ab = torch.clone(a).detach()
        ab[b > a] = b[b > a]
        return ab
