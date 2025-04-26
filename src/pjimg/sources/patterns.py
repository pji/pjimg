"""
Patterns
--------

Sources that generate shapes, text, and other fully deterministic
image and video data.

.. autoclass:: pjimg.sources.Box
.. autoclass:: pjimg.sources.Gradient
.. autoclass:: pjimg.sources.Hexes
.. autoclass:: pjimg.sources.Lines
.. autoclass:: pjimg.sources.Radials
.. autoclass:: pjimg.sources.Rays
.. autoclass:: pjimg.sources.Regular
.. autoclass:: pjimg.sources.Rings
.. autoclass:: pjimg.sources.Solid
.. autoclass:: pjimg.sources.Spheres
.. autoclass:: pjimg.sources.Spot
.. autoclass:: pjimg.sources.Text
.. autoclass:: pjimg.sources.Waves

"""
from math import pi, sqrt
from typing import Callable, Literal, Optional, Sequence, overload

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from pjimg.sources.model import Source
from pjimg.util import ImgAry, ImgTnsr, Loc, Size, X, Y, Z
from pjimg.util.tensor import IdxMeshes, hypot_3d, index_space
from pjimg.util.util import find_center


# Names available for import.
__all__ = [
    'Box', 'Gradient', 'Hexes', 'Lines', 'Radials', 'Rays', 'Regular',
    'Rings', 'Solid', 'Spheres', 'Spot', 'Text', 'Waves',
]


# Types.
WaveWarp = Callable[[ImgAry], ImgAry]
WaveWarpTnsr = Callable[[ImgTnsr], ImgTnsr]


# Public classes.
class Box(Source):
    """Draw a box.

    :param origin: The location of the upper left corner of the box.
    :param dimensions: The size of the box in three dimensions.
    :param color: The color of the box. This is a float within the
        range 0 <= x <= 1.
    :return: A :class:`Box` object.
    :rtype: sources.patterns.Box
    :usage:
        Create an image of a gray rectangle in the middle of a
        1280x720 image:

            >>> size = (1, 720, 1280)
            >>> origin = (n // 4 for n in size)
            >>> dimensions = (1, *(n // 2 for n in size[Y:]))
            >>> source = Box(origin=origin, dimensions=dimensions, color=0.5)
            >>> img = source.fill(size)

        .. figure:: images/box.jpg
           :alt: An image of a gray rectangle in the middle of a 1280x720
                 image.

           The image data created by the usage example.

    """
    def __init__(
        self, origin: Loc,
        dimensions: Size,
        color: float = 1.0,
        *args, **kwargs
    ) -> None:
        self.origin = origin
        self.dimensions = dimensions
        self.color = color
        super().__init__(*args, **kwargs)

    # Public methods.
    def fill_array(self, size: Size, loc: Loc = (0, 0, 0)) -> ImgAry:
        """Fill a volume with image data.

        :param size: The size of the volume of image data to generate.
        :param loc: (Optional.) How much to shift the starting point
            for the noise generation along each axis.
        :return: An :class:`numpy.ndarray` with image data.
        :rtype: numpy.ndarray
        """
        a = np.zeros(size)
        start = [n + o for n, o in zip(loc, self.origin)]
        end = [s + d for s, d in zip(start, self.dimensions)]
        slices = [slice(s, e) for s, e in zip(start, end)]
        a[tuple(slices)] = self.color
        return a

    def fill_tensor(self, size: Size, loc: Loc = (0, 0, 0)) -> ImgTnsr:
        """Fill a tensor with image data.

        :param size: The size of the volume of image data to generate.
        :param loc: (Optional.) How much to shift the starting point
            for the noise generation along each axis.
        :return: An :class:`torch.Tensor` with image data.
        :rtype: torch.Tensor
        """
        t = super().fill_tensor(size, loc)
        start = [n + o for n, o in zip(loc, self.origin)]
        end = [s + d for s, d in zip(start, self.dimensions)]
        slices = [slice(s, e) for s, e in zip(start, end)]
        t[tuple(slices)] = self.color
        return t


class Gradient(Source):
    """Generate a simple gradient.

    :param direction: (Optional.) This should be 'h' for a horizontal
        gradient or 'v' for a vertical gradient.
    :param stops: (Optional.) A gradient stop sets the color at a
        position in the gradient. This is a one-dimensional sequence
        of numbers. It's parsed in pairs, with the first number being
        the position of the stop and the second being the color value
        of the stop.
    :return: :class:`Gradient` object.
    :rtype: sources.patterns.Gradient
    :usage:
        Create a horizontal gradient with multiple stops in a
        1280x720 image.

            >>> size = (1, 720, 1280)
            >>> stops = [
            ...     0.1, 0.0,
            ...     0.2, 1.0,
            ...     0.3, 0.0,
            ...     0.8, 1.0,
            ...     0.9, 0.0,
            ... ]
            >>> source = Gradient(direction='h', stops=stops)
            >>> img = source.fill(size)

        .. figure:: images/gradient.jpg
           :alt: A horizontal gradient with multiple stops in a 1280x720
                image.

           The image data created by the usage example.

    """
    def __init__(
        self, direction: str = 'h',
        stops: Sequence[float] = (0, 0, 1, 1),
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.direction = direction

        # Parse the stops for the gradient.
        if isinstance(stops, str):
            stops = stops.split(',')
        self.stops = []
        for index in range(len(stops))[::2]:
            try:
                stop = [float(stops[index]), float(stops[index + 1])]
            except IndexError:
                msg = 'Missing color value for gradient stop.'
                raise ValueError(msg)
            self.stops.append(stop)

        # If the stops don't start at index zero, add a stop for
        # index zero to make the color between zero and the first
        # stop match the color at the first stop.
        if self.stops[0][0] != 0:
            self.stops.insert(0, [0, self.stops[0][1]])

        # If the stops don't end at index one, add a stop for index
        # one to make the color between the last stop and one match
        # the color of the last stop.
        if self.stops[-1][0] != 1:
            self.stops.append([1, self.stops[-1][1]])

    # Public methods.
    def fill_array(self, size: Size, loc: Loc = (0, 0, 0)) -> ImgAry:
        """Fill a volume with image data.

        :param size: The size of the volume of image data to generate.
        :param loc: (Optional.) How much to shift the starting point
            for the noise generation along each axis.
        :return: An :class:`numpy.ndarray` with image data.
        :rtype: numpy.ndarray
        """
        # Map out the locations of the stops within the gradient.
        if self.direction == 'h':
            a_size = size[X]
        elif self.direction == 'v':
            a_size = size[Y]
        elif self.direction == 't':
            a_size = size[Z]
        a = np.indices((a_size,))[0] / (a_size - 1)
        a_rev = 1 - a.copy()
        a_index = a.copy()

        # Interpolate the color values between the stops.
        # To do this I need to know the percentage of distance each
        # pixel represents. So, I need to do this in pairs.
        left_stop = self.stops[0]
        for right_stop in self.stops[1:]:

            # Create an array mask that isolates the area between the
            # two stops.
            mask = np.zeros(a.shape, bool)
            mask[a_index >= left_stop[0]] = True
            mask[a_index > right_stop[0]] = False

            # Determine where each pixel is within the area between
            # the two stops.
            distance = right_stop[0] - left_stop[0]
            a[mask] = a_index[mask] - left_stop[0]
            a[mask] = a[mask] / distance
            a_rev[mask] = 1 - a[mask]

            # Interpolate the color of the pixel based on its distance
            # from each of those stops and the color of those stops.
            a[mask] = a[mask] * right_stop[1]
            a_rev[mask] = a_rev[mask] * left_stop[1]
            a[mask] = a[mask] + a_rev[mask]

            # The right stop for this part of the gradient is the left
            # stop for the next part of the gradient.
            left_stop = right_stop

        # Run the easing function on the values and return the result.
        if self.direction == 'h':
            a = a.reshape(1, 1, a_size)
            a = np.tile(a, (size[Z], size[Y], 1))
        elif self.direction == 'v':
            a = a.reshape(1, a_size, 1)
            a = np.tile(a, (size[Z], 1, size[X]))
        elif self.direction == 't':
            a = a.reshape(a_size, 1, 1)
            a = np.tile(a, (1, size[Y], size[X]))
        return a

    def fill_tensor(self, size: Size, loc: Loc = (0, 0, 0)) -> ImgTnsr:
        """Fill a tensor with image data.

        :param size: The size of the volume of image data to generate.
        :param loc: (Optional.) How much to shift the starting point
            for the noise generation along each axis.
        :return: An :class:`torch.Tensor` with image data.
        :rtype: torch.Tensor
        """
        # Map out the locations of the stops within the gradient.
        if self.direction == 'h':
            t_size = size[X]
        elif self.direction == 'v':
            t_size = size[Y]
        elif self.direction == 't':
            t_size = size[Z]

        t = torch.arange(t_size, device=self.device) / (t_size - 1)
        t_index = torch.clone(t).detach()
        t_rev = 1 - t_index

        # Interpolate the color values between the stops.
        # To do this I need to know the percentage of distance each
        # pixel represents. So, I need to do this in pairs.
        left_stop = self.stops[0]
        for right_stop in self.stops[1:]:

            # Create an array mask that isolates the area between the
            # two stops.
            mask = torch.zeros(t.shape, dtype=torch.bool, device=self.device)
            mask[t_index >= left_stop[0]] = True
            mask[t_index > right_stop[0]] = False

            # Determine where each pixel is within the area between
            # the two stops.
            distance = right_stop[0] - left_stop[0]
            t[mask] = t_index[mask] - left_stop[0]
            t[mask] = t[mask] / distance
            t_rev[mask] = 1 - t[mask]

            # Interpolate the color of the pixel based on its distance
            # from each of those stops and the color of those stops.
            t[mask] = t[mask] * right_stop[1]
            t_rev[mask] = t_rev[mask] * left_stop[1]
            t[mask] = t[mask] + t_rev[mask]

            # The right stop for this part of the gradient is the left
            # stop for the next part of the gradient.
            left_stop = right_stop

        # Run the easing function on the values and return the result.
        if self.direction == 'h':
            t = t.reshape(1, 1, t_size)
            t = torch.tile(t, (size[Z], size[Y], 1))
        elif self.direction == 'v':
            t = t.reshape(1, t_size, 1)
            t = torch.tile(t, (size[Z], 1, size[X]))
        elif self.direction == 't':
            t = t.reshape(t_size, 1, 1)
            t = torch.tile(t, (1, size[Y], size[X]))
        return t


class Hexes(Source):
    """Fill a space with hexagonal cells.

    :param radius: The distance from the center of a cell to the center
        of each of the sides of the cell.
    :param cells: (Optional.) Whether the color of a pixel is based
        only on the distance to the nearest center point or if it
        is set to black if it's further away than the radius. When
        true, :class:`Hexes` will produce square cell-like structures
        rather than spheres.
    :param round: (Optional.) Whether to apply a circular easing
        function to output to give the appearance of the exterior
        of a sphere.
    :return: A :class:`sources.Hexes` object.
    :rtype: sources.patterns.Hexes
    :usage:
        Create a hexagonal grid of cells in a 1280x720 image.

            >>> size = (1, 720, 1280)
            >>> radius = size[Y] // 8
            >>> source = Hexes(radius=radius)
            >>> img = source.fill(size)

        .. figure:: images/hexes.jpg
           :alt: A hexagonal grid of cells in a 1280x720 image.

           The image data created by the usage example.

    """
    def __init__(
        self, radius: int,
        cells: bool = True,
        round: bool = False,
        *args, **kwargs
    ) -> None:
        self.cells = cells
        self.radius = radius
        self.round = round
        super().__init__(*args, **kwargs)

    # Public methods.
    def fill_array(self, size: Size, loc: Loc = (0, 0, 0)) -> ImgAry:
        """Fill a volume with image data.

        :param size: The size of the volume of image data to generate.
        :param loc: (Optional.) How much to shift the starting point
            for the noise generation along each axis.
        :return: An :class:`numpy.ndarray` with image data.
        :rtype: numpy.ndarray
        """
        # Place the centers of the hexagons.
        seeds = []
        x, y, row = 0.0, 0.0, 0.0
        xstep: float = self.radius
        ystep: float = sqrt(xstep ** 2 - (xstep / 2) ** 2)
        while y <= size[Y] + ystep:
            while x <= size[X] + xstep:
                seeds.append((y, x))
                x += xstep
            y += ystep
            row += 1
            x = 0
            if row % 2:
                x += xstep / 2

        # Map the distances to the points.
        indices = np.indices(size[1:])
        max_dist = np.sqrt(sum(n ** 2 for n in size))
        dist = np.zeros(size[1:], dtype=float)
        dist.fill(max_dist)
        for seed in seeds:
            axis_dist = [p - i for p, i in zip(seed, indices)]
            work = np.sqrt(sum(d ** 2 for d in axis_dist))
            dist[work < dist] = work[work < dist]
        act_max_dist = np.max(dist)
        a = dist / act_max_dist
        a = np.tile(a, (size[Z], 1, 1))
        if not self.cells:
            a[a > self.radius] = self.radius
        if self.round:
            a = np.sqrt(1 - a ** 2)
        else:
            a = 1 - a
        return a

    def fill_tensor(self, size: Size, loc: Loc = (0, 0, 0)) -> ImgTnsr:
        """Fill a tensor with image data.

        :param size: The size of the volume of image data to generate.
        :param loc: (Optional.) How much to shift the starting point
            for the noise generation along each axis.
        :return: An :class:`torch.Tensor` with image data.
        :rtype: torch.Tensor
        """
        # Place the centers of the hexagons.
        seeds = []
        x, y, row = 0.0, 0.0, 0.0
        xstep: float = self.radius
        ystep: float = sqrt(xstep ** 2 - (xstep / 2) ** 2)
        while y <= size[Y] + ystep:
            while x <= size[X] + xstep:
                seeds.append((y, x))
                x += xstep
            y += ystep
            row += 1
            x = 0
            if row % 2:
                x += xstep / 2

        # Map the distances to the points.
        indices = index_space(size[1:], device=self.device)
        max_dist = sqrt(sum(n ** 2 for n in size))
        dist = torch.zeros(size[1:], dtype=torch.float32, device=self.device)
        dist.fill_(max_dist)
        for seed in seeds:
            work = torch.hypot(*[p - i for p, i in zip(seed, indices)])
            dist[work < dist] = work[work < dist]
        act_max_dist = torch.max(dist)
        t = dist / act_max_dist
        t = torch.tile(t, (size[Z], 1, 1))
        if not self.cells:
            t[t > self.radius] = self.radius
        if self.round:
            t = torch.sqrt(1 - t ** 2)
        else:
            t = 1 - t
        return t


class Lines(Source):
    """Generate simple lines.

    :param direction: (Optional.) This should be 'h' for a horizontal
        gradient or 'v' for a vertical gradient.
    :param length: (Optional.) The distance between each line. Note:
        This parameter is hostile to proportional resizing. This is
        because one is subtracted from it when determining the period
        of the line. To allow for proportional resizing, add one to
        the value before passing it to this parameter.
    :return: :class:`Lines` object.
    :rtype: sources.patterns.Lines
    :usage:
        Create a series of vertical lines in a 1280x720 image.

            >>> size = (1, 720, 1280)
            >>> length = (size[X] / 8) - (size[Y] / 64)
            >>> source = Lines(direction='v', length=length)
            >>> img = source.fill(size)

        .. figure:: images/lines.jpg
           :alt: A picture of an image created from the output of
                :class:`Lines`.

           The image data created by the usage example.

    """
    def __init__(
        self, direction: str = 'h',
        length: float = 64,
        *args, **kwargs
    ) -> None:
        self.direction = direction
        self.length = float(length)
        super().__init__(*args, **kwargs)

    # Public methods.
    def fill_array(self, size: Size, loc: Loc = (0, 0, 0)) -> ImgAry:
        """Fill a volume with image data.

        :param size: The size of the volume of image data to generate.
        :param loc: (Optional.) How much to shift the starting point
            for the noise generation along each axis.
        :return: An :class:`numpy.ndarray` with image data.
        :rtype: numpy.ndarray
        """
        values = np.indices(size, dtype=float)
        values = shift_index_origin(values, loc)
        if self.direction == 'v':
            values = values[X] + values[Z]
        elif self.direction == 'h':
            values = values[Y] + values[Z]
        else:
            values = values[X] + values[Y]
        period = (self.length - 1)
        values = values % period
        values[values > period / 2] = period - values[values > period / 2]
        values = (values / (period / 2))
        return values.astype(np.float64)

    def fill_tensor(self, size: Size, loc: Loc = (0, 0, 0)) -> ImgTnsr:
        """Fill a tensor with image data.

        :param size: The size of the volume of image data to generate.
        :param loc: (Optional.) How much to shift the starting point
            for the noise generation along each axis.
        :return: An :class:`torch.Tensor` with image data.
        :rtype: torch.Tensor
        """
        meshes = index_space(
            size,
            loc,
            dtype=torch.float32,
            device=self.device
        )
        if self.direction == 'v':
            values = meshes[X] + meshes[Z]
        elif self.direction == 'h':
            values = meshes[Y] + meshes[Z]
        else:
            values = meshes[X] + meshes[Y]

        period = (self.length - 1)
        values = values % period
        values[values > period / 2] = period - values[values > period / 2]
        values = (values / (period / 2))
        return values.to(torch.float32)


class Radials(Source):
    """Generates concentric radial gradients.

    :param length: The radius of the innermost circle.
    :param growth: (Optional.) Either the string 'linear' or the
        string 'geometric'. Determines whether the distance between
        each circle remains constant (linear) or increases
        (geometric). Defaults to linear.
    :returns: :class:`Radials` object.
    :rtype: sources.patterns.Radials
    :usage:
        Create a series of concentric radial gradients in a

            >>> # 1280x720 images.
            >>> size = (1, 720, 1280)
            >>> length = size[Y] / 16
            >>> source = Radials(length=length, growth='g')
            >>> img = source.fill(size)

        .. figure:: images/radials.jpg
           :alt: A series of concentric radial gradients in a 1280x720 image.

           The image data created by the usage example.

    """
    def __init__(
        self, length: float,
        growth: str = 'l',
        *args, **kwargs
    ) -> None:
        """Initialize an instance of Radials."""
        self.length = float(length)
        self.growth = growth
        super().__init__(*args, **kwargs)

    # Public methods.
    def fill_array(self, size: Size, loc: Loc = (0, 0, 0)) -> ImgAry:
        """Fill a volume with image data.

        :param size: The size of the volume of image data to generate.
        :param loc: (Optional.) How much to shift the starting point
            for the noise generation along each axis.
        :return: An :class:`numpy.ndarray` with image data.
        :rtype: numpy.ndarray
        """
        # Map out the volume of space that will be created.
        a = np.zeros(size, dtype=float)
        c = np.indices(size, dtype=float)
        c = center_index_origin(c)
        c = shift_index_origin(c, loc)

        # Perform a spherical interpolation on the points in the
        # volume and run the easing function on the results.
        c = index_to_distance_from_origin(c)
        if self.growth == 'l' or self.growth == 'linear':
            a = c % self.length
            a /= self.length
            a = abs(a - .5) * 2

        elif self.growth == 'g' or self.growth == 'geometric':
            in_length = 0.0
            out_length = self.length
            while in_length < np.max(c):
                m = np.ones(a.shape, bool)
                m[c < in_length] = False
                m[c > out_length] = False
                a[m] = c[m]
                a[m] -= in_length
                a[m] /= out_length - in_length
                a[m] = abs(a[m] - .5) * 2
                in_length = out_length
                out_length *= 2

        return a

    def fill_tensor(self, size: Size, loc: Loc = (0, 0, 0)) -> ImgTnsr:
        """Fill a tensor with image data.

        :param size: The size of the volume of image data to generate.
        :param loc: (Optional.) How much to shift the starting point
            for the noise generation along each axis.
        :return: An :class:`torch.Tensor` with image data.
        :rtype: torch.Tensor
        """
        # Map out the volume of space that will be created.
        meshes = index_space(
            size,
            loc,
            dtype=torch.float32,
            device=self.device,
            center=True
        )

        # Perform a spherical interpolation on the points in the
        # volume and run the easing function on the results.
        c = torch.hypot(meshes[X], meshes[Y])
        if self.growth == 'l' or self.growth == 'linear':
            t = c % self.length
            t /= self.length
            t = torch.abs(t - .5) * 2

        elif self.growth == 'g' or self.growth == 'geometric':
            t = super().fill_tensor(size, loc)
            in_length = 0.0
            out_length = self.length
            while in_length < torch.max(c):
                m = torch.ones(t.shape, dtype=torch.bool, device=self.device)
                m[c < in_length] = False
                m[c > out_length] = False
                t[m] = c[m]
                t[m] -= in_length
                t[m] /= out_length - in_length
                t[m] = torch.abs(t[m] - .5) * 2
                in_length = out_length
                out_length *= 2

        return t


class Rays(Source):
    """Create rays that generate from a central point.

    :param count: The number of rays to generate.
    :param offset: (Optional.) Rotate the rays around the generation
        point. This is measured in radians.
    :return: :class:`Rays` object.
    :rtype: sources.patterns.Rays

    Usage::

        >>> # Create seven rays emanating from the center of a 1280x720
        >>> # image.
        >>> size = (1, 720, 1280)
        >>> source = Rays(count=7, offset=0.178)
        >>> img = source.fill(size)

    .. figure:: images/rays.jpg
       :alt: Seven rays emanating from the center of a 1280x720 image.

       The image data created by the usage example.

    """
    def __init__(
        self, count: int,
        offset: float = 0,
        *args, **kwargs
    ) -> None:
        self.count = int(count)
        self.offset = float(offset)
        super().__init__(*args, **kwargs)

    # Public methods.
    def fill_array(self, size: Size, loc: Loc = (0, 0, 0)) -> ImgAry:
        """Fill a volume with image data.

        :param size: The size of the volume of image data to generate.
        :param loc: (Optional.) How much to shift the starting point
            for the noise generation along each axis.
        :return: An :class:`numpy.ndarray` with image data.
        :rtype: numpy.ndarray
        """
        # Determine the angle from center for every point
        # in the array.
        indices = np.indices(size, dtype=float)
        indices = center_index_origin(indices)
        indices = shift_index_origin(indices, loc)
        x, y = indices[X], indices[Y]
        angle = np.zeros_like(x)
        angle[x != 0] = np.arctan(y[x != 0] / x[x != 0])

        # Correct for inaccuracy of the arctan function when one or
        # both of the coordinates is less than zero.
        m = np.zeros_like(x)
        m[x < 0] += 1
        m[y < 0] += 3
        angle[m == 1] += np.pi
        angle[m == 4] += np.pi
        angle[m == 3] += 2 * np.pi

        # Create the rays.
        ray_angle = 2 * np.pi / self.count
        offset = (self.offset * np.pi) % (2 * np.pi)
        rays = (angle + offset) % ray_angle
        rays /= ray_angle
        rays = abs(rays - .5) * 2

        # Fill in the center if needed.
        center = [(n - 1) / 2 + o for n, o in zip(size, loc)]
        if center[X] % 1 == 0 and center[Y] % 1 == 0:
            center = [int(n) for n in center]
            rays[(center[Y], center[X])] = 1

        # Fill out the Z axis and return.
        rays = np.tile(rays, (size[Z], 1, 1))
        return rays

    def fill_tensor(self, size: Size, loc: Loc = (0, 0, 0)) -> ImgTnsr:
        """Fill a tensor with image data.

        :param size: The size of the volume of image data to generate.
        :param loc: (Optional.) How much to shift the starting point
            for the noise generation along each axis.
        :return: An :class:`torch.Tensor` with image data.
        :rtype: torch.Tensor
        """
        # Determine the angle from center for every point
        # in the array.
        meshes = index_space(
            size,
            loc,
            dtype=torch.float32,
            device=self.device,
            center=True
        )
        x, y = meshes[X], meshes[Y]
        angle = torch.zeros_like(x)
        angle[x != 0] = torch.arctan(y[x != 0] / x[x != 0])

        # Correct for inaccuracy of the arctan function when one or
        # both of the coordinates is less than zero.
        m = torch.zeros_like(x)
        m[x < 0] += 1
        m[y < 0] += 3
        angle[m == 1] += pi
        angle[m == 4] += pi
        angle[m == 3] += 2 * pi

        # Create the rays.
        ray_angle = 2 * pi / self.count
        offset = (self.offset * pi) % (2 * pi)
        rays = (angle + offset) % ray_angle
        rays /= ray_angle
        rays = torch.abs(rays - .5) * 2

        # Fill in the center if needed.
        center = [(n - 1) / 2 + o for n, o in zip(size, loc)]
        if center[X] % 1 == 0 and center[Y] % 1 == 0:
            rays[int(center[Y]), int(center[X])] = 1

        # Fill out the Z axis and return.
        rays = torch.tile(rays, (size[Z], 1, 1))
        return rays


class Regular(Source):
    """Create a regular polygon.

    :param sides: The number of sides of the polygon.
    :param rho: The distance from the center of the polygon to
        a vertex of the polygon.
    :param rotate: (Optional.) How much to rotate the polygon in
        radians. The default is `0.0`.
    :param color: (Optional.) The color of the polygon. Default is
        `1.0`.
    :param antialias: (Optional.) Whether to antialias the edge
        of the polygon. Default is `True`.
    :usage:
        Create a pentagon in a 1280x720 image.

            >>> size = (1, 720, 1280)
            >>> source = Regular(5, size[1] / 2)
            >>> img = source.fill(size)

        .. figure:: images/regular.jpg
           :alt: A pentagon in the center of a 1280x720 image.

           The image data created by the usage example.

    """
    def __init__(
        self, sides: int,
        rho: float,
        rotate: float = 0.0,
        color: float = 1.0,
        bg_color: float = 0.0,
        antialias: bool = False,
        *args, **kwargs
    ) -> None:
        self._disallowed_devices = ['mps',]
        super().__init__(*args, **kwargs)
        self.sides = sides
        self.rho = rho
        self.rotate = rotate
        self.color = color
        self.bg_color = bg_color
        self.antialias = antialias

    def fill_array(self, size: Size, loc: Loc = (0, 0, 0)) -> ImgAry:
        """Fill a volume with image data.

        :param size: The size of the volume of image data to generate.
        :param loc: (Optional.) How much to shift the starting point
            for the noise generation along each axis.
        :return: An :class:`numpy.ndarray` with image data.
        :rtype: numpy.ndarray
        """
        center = [n // 2 + o for n, o in zip(size, loc)]
        angle = 2 * np.pi / self.sides
        rho = self.rho
        start = (3 * np.pi / 2 + self.rotate) % (2 * np.pi)
        vertices = np.array([[
            (
                rho * np.cos(start + i * angle) + center[2],
                rho * np.sin(start + i * angle) + center[1],
            )
            for i in range(self.sides)
        ]], dtype=np.int32)

        a = np.zeros(size[1:], dtype=np.uint8)
        bg_color = int(self.bg_color * 255)
        a.fill(bg_color)
        color = int(self.color * 255)
        line_type = cv2.LINE_8
        if self.antialias:
            line_type = cv2.LINE_AA
        cv2.fillConvexPoly(a, vertices, color=(color,), lineType=line_type)
        a = a[np.newaxis, :, :]
        a = np.tile(a, (size[Z], 1, 1))
        return a.astype(float) / 255

    def fill_tensor(self, size: Size, loc: Loc = (0, 0, 0)) -> ImgTnsr:
        """Fill a tensor with image data.

        :param size: The size of the volume of image data to generate.
        :param loc: (Optional.) How much to shift the starting point
            for the noise generation along each axis.
        :return: An :class:`torch.Tensor` with image data.
        :rtype: torch.Tensor
        """
        # Calculate the vertices.
        center = [n // 2 + o for n, o in zip(size, loc)]
        angle = 2 * pi / self.sides
        rho = self.rho
        start = (3 * pi / 2 + self.rotate) % (2 * pi)
        vertices = torch.tensor([[
            (
                rho * np.cos(start + i * angle) + center[2],
                rho * np.sin(start + i * angle) + center[1],
            )
            for i in range(self.sides)
        ]], dtype=torch.int32, device=self.device)

        # Make the background.
        t = torch.full(
            size[1:],
            fill_value=int(self.bg_color * 255),
            dtype=torch.uint8,
            device=self.device
        )

        # Make the shape.
        color = int(self.color * 255)
        line_type = cv2.LINE_8
        if self.antialias:
            line_type = cv2.LINE_AA
        cv2.fillConvexPoly(
            t.numpy(),
            vertices.numpy(),
            color=(color,),
            lineType=line_type
        )
        t = t.unsqueeze(0)
        t = torch.tile(t, (size[Z], 1, 1))
        return t.to(torch.float32) / 255


class Rings(Source):
    """Create a series of concentric circles.

    :param radius: The radius of the first ring, which is the ring
        closest to the center. It is measured from the origin point
        of the rings to the middle of the band of the first ring.
    :param width: The width of each band of the ring. It's measured
        from each edge of the band.
    :param gap: (Optional.) The distance between each ring. It's
        measured from the middle of the first band to the middle
        of the next band. The default value of zero causes the
        rings to draw on top of each other, making it look like
        there is only one ring.
    :param count: (Optional.) The number of rings to draw. The
        default is one.
    :param count_offset: (Optional.) Set to `1` for backwards
        compatibility with :mod:`pjinoise`.
    :return: :class:`Rings` object.
    :rtype: sources.patterns.Rings
    :usage:
        Create a series of concentric rings in a 1280x720 image.

            >>> size = (1, 720, 1280)
            >>> radius = size[X] / 6
            >>> width = size[X] / 12
            >>> gap = size[X] / 18
            >>> ct = 6
            >>> source = Rings(radius=radius, width=width, gap=gap, count=ct)
            >>> img = source.fill(size)

        .. figure:: images/rings.jpg
           :alt: A series of concentric rings in a 1280.720 image.

           The image data created by the usage example.

    """
    def __init__(
        self, radius: float,
        width: float,
        gap: float = 0,
        count: int = 1,
        count_offset: int = 0,
        *args, **kwargs
    ) -> None:
        """Initialize an instance of Ring."""
        super().__init__(*args, **kwargs)
        self.radius = float(radius)
        self.width = float(width)
        self.gap = float(gap)
        self.count = int(count)
        self.count_offset = count_offset

    # Public methods.
    def fill_array(self, size: Size, loc: Loc = (0, 0, 0)) -> ImgAry:
        """Fill a volume with image data.

        :param size: The size of the volume of image data to generate.
        :param loc: (Optional.) How much to shift the starting point
            for the noise generation along each axis.
        :return: An :class:`numpy.ndarray` with image data.
        :rtype: numpy.ndarray
        """
        # Map out the volume of space that will be created.
        a = np.zeros(size, dtype=float)
        c = np.indices(size, dtype=float)
        c = center_index_origin(c)
        c = shift_index_origin(c, loc)

        # Perform a spherical interpolation on the points in the
        # volume and run the easing function on the results.
        c = index_to_distance_from_origin(c)
        for i in range(self.count):
            radius = self.radius + self.gap * (i - self.count_offset)
            if radius != 0:
                working = c / np.sqrt(radius ** 2)
                working = np.abs(working - 1)
                wr = self.width / 2 / radius
                m = np.zeros(working.shape, bool)
                m[working <= wr] = True
                a[m] = working[m] * (radius / (self.width / 2))
                a[m] = 1 - a[m]
        return a

    def fill_tensor(self, size: Size, loc: Loc = (0, 0, 0)) -> ImgTnsr:
        """Fill a tensor with image data.

        :param size: The size of the volume of image data to generate.
        :param loc: (Optional.) How much to shift the starting point
            for the noise generation along each axis.
        :return: An :class:`torch.Tensor` with image data.
        :rtype: torch.Tensor
        """
        # Map out the volume of space that will be created.
        meshes = index_space(
            (1, *size[Y:]),
            loc,
            device=self.device,
            dtype=torch.float32,
            center=True
        )

        # Perform a spherical interpolation on the points in the
        # volume and run the easing function on the results.
        c = torch.hypot(meshes[Y], meshes[X])
        t = super().fill_tensor((1, *size[Y:]), loc)
        for i in range(self.count):
            radius = self.radius + self.gap * (i - self.count_offset)
            if radius != 0:
                working = torch.abs(c / radius - 1)
                wr = self.width / 2 / radius
                m = torch.zeros(
                    working.shape,
                    dtype=torch.bool,
                    device=self.device
                )
                m[working <= wr] = True
                t[m] = 1 - (working[m] * (radius / (self.width / 2)))
        return torch.tile(t, (size[Z], 1, 1))


class Solid(Source):
    """Fill a space with a solid color.

    :param color: The color to use for the fill. Zero is black. One
        is white. The values between are values of gray.
    :return: :class:`Solid` object.
    :rtype: pjinoise.sources.Solid
    :usage:
        Create a solid gray 1280x720 image.

            >>> size = (1, 1280, 720)
            >>> source = Solid(color=0.25)
            >>> img = source.fill(size)

        .. figure:: images/solid.jpg
           :alt: A solid gray 1280x720 image.

           The image data created by the usage example.

    """
    def __init__(self, color: float, *args, **kwargs) -> None:
        self.color = float(color)
        super().__init__(*args, **kwargs)

    # Public methods.
    def fill_array(self, size: Size, loc: Loc = (0, 0, 0)) -> ImgAry:
        """Fill a volume with image data.

        :param size: The size of the volume of image data to generate.
        :param loc: (Optional.) How much to shift the starting point
            for the noise generation along each axis.
        :return: An :class:`numpy.ndarray` with image data.
        :rtype: numpy.ndarray
        """
        a = np.zeros(size, dtype=float)
        a.fill(self.color)
        return a

    def fill_tensor(self, size: Size, loc: Loc = (0, 0, 0)) -> ImgTnsr:
        """Fill a tensor with image data.

        :param size: The size of the volume of image data to generate.
        :param loc: (Optional.) How much to shift the starting point
            for the noise generation along each axis.
        :return: An :class:`numpy.ndarray` with image data.
        :rtype: numpy.ndarray
        """
        return torch.full(
            size,
            self.color,
            dtype=torch.float32,
            device=self.device
        )


class Spheres(Source):
    """Fill a space with a series of spots.

    :param radius: The radius of an individual spot.
    :param offset: (Optional.) Whether alternating rows or columns
        should be offset. Set to 'x' for rows to be offset. Set to
        'y' for columns to be offset. It defaults to None for no
        offset.
    :param cells: (Optional.) Whether the color of a pixel is based
        only on the distance to the nearest center point or if it
        is set to black if it's further away than the radius. When
        true, :class:`Spheres` will produce square cell-like structures
        rather than spheres.
    :param round: (Optional.) Whether to apply a circular easing
        function to output to give the appearance of the exterior
        of a sphere.
    :return: :class:`Spheres` object.
    :rtype: sources.patterns.Spheres
    :usage:
        Create a square grid of cells in a 1280x720 image.

            >>> size = (1, 720, 1280)
            >>> radius = size[Y] / 16
            >>> source = Spheres(radius=radius, offset='')
            >>> img = source.fill(size)

        .. figure:: images/spheres.jpg
           :alt: A square grid of cells in a 1280x720 image.

           The image data created by the usage example.

    """
    def __init__(
        self, radius: float,
        offset: str = '',
        cells: bool = False,
        round: bool = True,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.cells = cells
        self.offset = offset
        self.radius = float(radius)
        self.round = round

    # Public methods.
    def fill_array(self, size: Size, loc: Loc = (0, 0, 0)) -> ImgAry:
        """Fill a volume with image data.

        :param size: The size of the volume of image data to generate.
        :param loc: (Optional.) How much to shift the starting point
            for the noise generation along each axis.
        :return: An :class:`numpy.ndarray` with image data.
        :rtype: numpy.ndarray
        """
        # Map out the volume of space that will be created.
        a = np.indices(size, dtype=float)
        a = shift_index_origin(a, loc)

        # If configured, offset every other row, column, or plane by
        # by the radius of the circle.
        if self.offset == 'x':
            mask = np.zeros(a[Y].shape, bool)
            d = self.radius * 2
            dd = d * 2
            mask[a[Y] % dd < d] = True

            # Note: This used to be just subtracting the radius from
            # a[X][~mask], but it stopped working. I'm not sure why.
            # Maybe it never did, and my headache was keeping me from
            # noticing it. Either way, this seems to work.
            a[X][mask] = a[X][mask] + self.radius
            a[Y][mask] = a[Y][mask] + self.radius
            a[Y][~mask] = a[Y][~mask] + self.radius

        if self.offset == 'y':
            mask = np.zeros(a[X].shape, bool)
            d = self.radius * 2
            dd = d * 2
            mask[a[X] % dd < d] = True

            # Note: For some reason, this is not the same as just
            # subtracting the radius from a[Y][mask]. I don't know
            # why, and my headache is making me disinclined to look
            # at the math.
            a[X][mask] = a[X][mask] + self.radius
            a[X][~mask] = a[X][~mask] + self.radius
            a[Y][~mask] = a[Y][~mask] + self.radius

        # Split the volume into unit cubes that are the size of the
        # diameter of the circle. Then adjust the indicies to measure
        # the distance to the nearest unit rather than the distance
        # from the last unit.
        a = a % (self.radius * 2)
        a[a > self.radius] = self.radius * 2 - a[a > self.radius]

        # Interpolate the unit distances through the sphere equation
        # to generate the regularly spaced spheres in the volume.
        # Then run the easing function on those spheres.
        a = np.sqrt(a[X] ** 2 + a[Y] ** 2 + a[Z] ** 2)
        if self.cells:
            a = (a / np.sqrt(3 * self.radius ** 2))
        else:
            a[a > self.radius] = self.radius
            a /= self.radius
        if self.round:
            a = np.sqrt(1 - a ** 2)
        else:
            a = 1 - a
        return a

    def fill_tensor(self, size: Size, loc: Loc = (0, 0, 0)) -> ImgTnsr:
        """Fill a tensor with image data.

        :param size: The size of the volume of image data to generate.
        :param loc: (Optional.) How much to shift the starting point
            for the noise generation along each axis.
        :return: An :class:`numpy.ndarray` with image data.
        :rtype: numpy.ndarray
        """
        # Map out the volume of space that will be created.
        meshes = index_space(
            size,
            loc,
            dtype=torch.float32,
            device=self.device
        )
        meshes = list(meshes)

        # If configured, offset every other row, column, or plane by
        # by the radius of the circle.
        if self.offset == 'x':
            mask = torch.zeros(
                meshes[Y].shape,
                dtype=torch.bool,
                device=self.device
            )
            d = self.radius * 2
            dd = d * 2
            mask[meshes[Y] % dd < d] = True

            # Create clones of the index meshes to avoid a deprecation
            # message from torch.
            x = meshes[X].clone().detach()
            y = torch.clone(meshes[Y]).detach()

            # Note: This used to be just subtracting the radius from
            # a[X][~mask], but it stopped working. I'm not sure why.
            # Maybe it never did, and my headache was keeping me from
            # noticing it. Either way, this seems to work.
            x[mask] = meshes[X][mask] + self.radius
            y += self.radius

            # Replace the originals with the clones to avoid the
            # index_put_ warning from torch that occurs if we just
            # perform the operation directly on the original.
            meshes[X] = x
            meshes[Y] = y

        if self.offset == 'y':
            mask = torch.zeros(
                meshes[X].shape,
                dtype=torch.bool,
                device=self.device
            )
            d = self.radius * 2
            dd = d * 2
            mask[meshes[X] % dd < d] = True

            # Create clones of the index meshes to avoid a deprecation
            # message from torch.
            x = meshes[X].clone().detach()
            y = torch.clone(meshes[Y]).detach()

            # Note: For some reason, this is not the same as just
            # subtracting the radius from a[Y][mask]. I don't know
            # why, and my headache is making me disinclined to look
            # at the math.
            x += self.radius
            y[~mask] = meshes[Y][~mask] + self.radius

            # Replace the originals with the clones to avoid the
            # index_put_ warning from torch that occurs if we just
            # perform the operation directly on the original.
            meshes[X] = x
            meshes[Y] = y

        # Split the volume into unit cubes that are the size of the
        # diameter of the circle. Then adjust the indicies to measure
        # the distance to the nearest unit rather than the distance
        # from the last unit.
        meshes = [mesh % (self.radius * 2) for mesh in meshes]
        for m in meshes:
            m[m > self.radius] = self.radius * 2 - m[m > self.radius]

        # Interpolate the unit distances through the sphere equation
        # to generate the regularly spaced spheres in the volume.
        # Then run the easing function on those spheres.
        t = hypot_3d(meshes)
        if self.cells:
            t = (t / sqrt(3 * self.radius ** 2))
        else:
            t[t > self.radius] = self.radius
            t /= self.radius
        if self.round:
            t = torch.sqrt(1 - t ** 2)
        else:
            t = 1 - t
        return t


class Spot(Source):
    """Fill a space with a spot.

    :param radius: The radius of the spot.
    :return: :class:`Spot` object.
    :rtype: sources.patterns.Spot
    :usage:
        Create a radial gradient centered in a 1280x720 image.

            >>> size = (1, 720, 1280)
            >>> radius = size[Y] * 2 / 3
            >>> source = Spot(radius=radius)
            >>> img = source.fill(size)

        .. figure:: images/spot.jpg
           :alt: A radial gradient centered in a 1280x720 image.

           The image data created by the usage example.

    """
    def __init__(self, radius: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.radius = float(radius)

    # Public methods.
    def fill_array(self, size: Size, loc: Loc = (0, 0, 0)) -> ImgAry:
        """Fill a volume with image data.

        :param size: The size of the volume of image data to generate.
        :param loc: (Optional.) How much to shift the starting point
            for the noise generation along each axis.
        :return: An :class:`numpy.ndarray` with image data.
        :rtype: numpy.ndarray
        """
        # Map out the volume of space that will be created.
        a = np.indices(size, dtype=float)
        a = center_index_origin(a)
        a = shift_index_origin(a, loc)

        # Perform a spherical interpolation on the points in the
        # volume and run the easing function on the results.
        a = index_to_distance_from_origin(a)
        a = 1 - (a / np.sqrt(2 * self.radius ** 2))
        a[a > 1] = 1
        a[a < 0] = 0
        return a

    def fill_tensor(self, size: Size, loc: Loc = (0, 0, 0)) -> ImgTnsr:
        """Fill a tensor with image data.

        :param size: The size of the volume of image data to generate.
        :param loc: (Optional.) How much to shift the starting point
            for the noise generation along each axis.
        :return: An :class:`numpy.ndarray` with image data.
        :rtype: numpy.ndarray
        """
        # Map out the volume of space that will be created.
        *_, idx_y, idx_x = index_space(
            size=size,
            loc=loc,
            device=self.device,
            dtype=torch.float32,
            center=True
        )

        # Perform a spherical interpolation on the points in the
        # volume and run the easing function on the results.
        t = torch.hypot(idx_y, idx_x)
        t = 1 - (t / sqrt(2 * self.radius ** 2))
        t[t > 1] = 1
        t[t < 0] = 0
        return t


class Text(Source):
    """Place text within the image.

    :param text: The text to add.
    :param font: (Optional.) The font for the text. It uses the fonts
        available to your system.
    :param size: (Optional.) The size of the text in points.
    :param face: (Optional.) The index number of the face of the font. See
        :meth:`PIL.ImageFont.truetype`.
    :param encoding: (Optional.) The encoding for the font. See
        :meth:`PIL.ImageFont.truetype`.
    :param layout_engine: (Optional.) The layout engine for the font. See
        :meth:`PIL.ImageFont.truetype`.
    :param origin: (Optional.) The starting position for the test.
    :param start: (Optional.) The number of blank frames before the text
        appears in video.
    :param duration: (Optional.) The number of frames the texts is visible
        in video.
    :param fill_color: (Optional.) The brightness of the text.
    :param bg_color: (Optional.) The color of the background behind the
        text.
    :param spacing: (Optional.) The number of pixels between lines of text.
        Basically the leading minus the size.
    :param spacing: (Optional.) How to automatically calculate the spacing.
    :param align: (Optional.) The horizontal alignment of the text.
    :param stroke_width: (Optional.) The width of the stroke around the
        characters.
    :param stroke_color: (Optional.) The color to use for the stroke.
    :return: A :class:`Text` object.
    :rtype: sources.patterns.Text
    :usage:
        Create the word "SPAM" in a 1280x720 image.

            >>> size = (1, 720, 1280)
            >>> origin = (size[X] / 2 - 107, size[Y] / 2 - 52)
            >>> source = Text(
            ...     text='SPAM',
            ...     font='Helvetica',
            ...     size=72,
            ...     face=1,
            ...     layout_engine='basic',
            ...     origin=origin,
            ...     fill_color=0.75,
            ...     bg_color=0.25,
            ...     align='center',
            ...     stroke_width=5,
            ...     stroke_fill=0x00
            ... )
            >>> img = source.fill(size)

        .. figure:: images/text.jpg
           :alt: The word "SPAM" in a 1280x720 image.

           The image data created by the usage example.

    """
    def __init__(
        self, text: str,
        font: str = 'Verdana',
        size: int = 10,
        face: int = 0,
        encoding: str = 'unic',
        layout_engine: str = '',
        origin: tuple[float, float] = (0, 0),
        start: int = 0,
        duration: Optional[int] = None,
        fill_color: float = 1,
        bg_color: float = 0,
        spacing: float = .2,
        spacing_mode: str = 'proportional',
        align: Literal['left', 'center', 'right'] = 'left',
        stroke_width: int = 0,
        stroke_fill: int = 0,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.text = text
        self.font = font
        self.size = size
        self.face = face
        self.encoding = encoding

        if not layout_engine:
            self.layout_engine = None
        elif layout_engine == 'basic':
            self.layout_engine = ImageFont.Layout.BASIC
        elif layout_engine == 'raqm':
            self.layout_engine = ImageFont.Layout.RAQM
        else:
            msg = f'{layout_engine} is not a valid value for layout_engine.'
            raise ValueError(msg)

        self.origin = origin
        self.start = start
        self.duration = duration
        self.fill_color = int(fill_color * 0xff)
        self.bg_color = int(bg_color * 0xff)

        if spacing_mode == 'proportional' or spacing_mode == 'p':
            self.spacing = self.size * spacing
        else:
            self.spacing = spacing

        self.align = align
        self.stroke_width = stroke_width
        self.stroke_fill = stroke_fill

        self._font = ImageFont.truetype(
            self.font, self.size, self.face,
            self.encoding, self.layout_engine
        )

    # Public methods.
    def fill_array(self, size: Size, loc: Loc = (0, 0, 0)) -> ImgAry:
        """Fill a volume with image data.

        :param size: The size of the volume of image data to generate.
        :param loc: (Optional.) How much to shift the starting point
            for the noise generation along each axis.
        :return: An :class:`numpy.ndarray` with image data.
        :rtype: numpy.ndarray
        """
        a = np.zeros(size, float)
        origin = (
            self.origin[0] + loc[Y],
            self.origin[1] + loc[X],
        )
        start = self.start - loc[Z]
        if self.duration is None:
            end = size[Z]
        else:
            end = start + self.duration

        img = Image.new('L', (size[X], size[Y]), self.bg_color)
        draw = ImageDraw.Draw(img)
        draw.text(
            xy=origin,
            text=self.text,
            fill=self.fill_color,
            font=self._font,
            anchor=None,
            spacing=self.spacing,
            align=self.align,
            stroke_width=self.stroke_width,
            stroke_fill=self.stroke_fill
        )
        for i in range(a.shape[Z]):
            if i >= end:
                break
            if i >= start:
                a[i] = (np.array(img).astype(float) / 0xff)

        return a

    def fill_tensor(self, size: Size, loc: Loc = (0, 0, 0)) -> ImgTnsr:
        """Fill a volume with image data.

        :param size: The size of the volume of image data to generate.
        :param loc: (Optional.) How much to shift the starting point
            for the noise generation along each axis.
        :return: An :class:`numpy.ndarray` with image data.
        :rtype: numpy.ndarray
        """
        # Set up parameters for the text generation.
        origin = (
            self.origin[0] + loc[Y],
            self.origin[1] + loc[X],
        )
        start = self.start - loc[Z]
        end = size[Z]
        if self.duration is not None:
            end = start + self.duration

        # Generate the text.
        img = Image.new('L', (size[X], size[Y]), self.bg_color)
        draw = ImageDraw.Draw(img)
        draw.text(
            xy=origin,
            text=self.text,
            fill=self.fill_color,
            font=self._font,
            anchor=None,
            spacing=self.spacing,
            align=self.align,
            stroke_width=self.stroke_width,
            stroke_fill=self.stroke_fill
        )

        # Add the text to the frames and return.
        t = super().fill_tensor(size, loc)
        a = np.array(img, dtype=np.float32) / 0xff
        for frame in t[:end]:
            frame[:] = torch.tensor(a, device=self.device)
        return t


class Waves(Source):
    """Generates wave patterns using a cosine function.

    :param unit: (Optional.) The number of pixels in a unit of distance
        of the wave. It defaults to 1279 pixels.
    :param angle: (Optional.) The angle in degrees of the wave. An
        angle of 0 creates vertical bars. An angle of 90 creates
        horizontal bars. It defaults to 0.
    :param wavelength: (Optional.) The number peaks that occur within
        a unit. Defaults to 1.
    :param warp: (Optional.) A function that accepts an image
        array and returns an image array. This function is used to
        alter the space the wave is propagating through, allowing you
        to change the wavelength based on the position in the image.
        Defaults to `None`.
    :param radial: (Optional.) When true, the waves should radiate from
        a central point in a circular pattern. Defaults to `False`.
    :return: :class:`Waves` object.
    :rtype: sources.patterns.Waves
    :usage:
        Create a wave pattern in a 1280x720 image.

            >>> size = (1, 720, 1280)
            >>> unit = 1279
            >>> angle = 30.0
            >>> wavelength = 5.0
            >>> source = Waves(unit, angle=angle, wavelength=wavelength)
            >>> img = source.fill(size)

        .. figure:: images/waves.jpg
           :alt: Create a wave pattern in a 1280x720 image.

           The image data created by the usage example.

    """
    def __init__(
        self, unit: int = 1279,
        angle: float = 0,
        wavelength: float = 1,
        warp: Optional[WaveWarp] = None,
        warp_tensor: Optional[WaveWarpTnsr] = None,
        radial: bool = False,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.unit = unit
        self.angle = angle
        self.wavelength = wavelength
        self.warp = warp
        self.warp_tensor = warp_tensor
        self.radial = radial

    def fill_array(self, size: Size, loc: Loc = (0, 0, 0)) -> ImgAry:
        x = self.angle / 90
        indices = np.indices(size, dtype=float)
        indices = shift_index_origin(indices, loc)
        f = (2 * np.pi) / self.wavelength

        # Set the angle of the wave.
        if not self.radial:
            a = indices[X] * (1 - x) + indices[Y] * x

            # Factors in the Z axis for video. The pace of change
            # over the Z axis should probably be something that can
            # be set.
            if size[Z] > 1:
                a += indices[Z] * (self.unit / 48)

        else:
            indices = center_index_origin(indices)
            a = index_to_distance_from_origin(indices)

        # Break the grid into units.
        a /= self.unit

        # Modify how the wave evolves using the acceleration function.
        if self.warp:
            a = self.warp(a)

        # Return the wave.
        a = (np.cos(f * a) + 1) / 2
        return a

    def fill_tensor(self, size: Size, loc: Loc = (0, 0, 0)) -> ImgTnsr:
        x = self.angle / 90
        f = (2 * pi) / self.wavelength

        # Set the angle of the wave.
        if not self.radial:
            meshes = index_space(size, loc, self.device, torch.float32)
            t = meshes[X] * (1 - x) + meshes[Y] * x

            # Factors in the Z axis for video. The pace of change
            # over the Z axis should probably be something that can
            # be set.
            if size[Z] > 1:
                t += meshes[Z] * (self.unit / 48)

        else:
            meshes = index_space(size, loc, self.device, torch.float32, True)
            t = torch.hypot(meshes[Y], meshes[X])

        # Break the grid into units.
        t /= self.unit

        # Modify how the wave evolves using the acceleration function.
        if self.warp_tensor:
            t = self.warp_tensor(t)
        elif self.warp:
            a = self.warp(t.numpy())
            t = torch.tensor(a)

        # Return the wave.
        return (torch.cos(f * t) + 1) / 2


# Utility functions.
def center_index_origin(indices: ImgAry) -> ImgAry:
    """Move the origin point of an indices array to the center of
    the array.
    """
    size = indices[0].shape
    shifts = [(n - 1) / 2 for n in size]
    return shift_index_origin(indices, shifts)


def index_to_distance_from_origin(indices: ImgAry) -> ImgAry:
    """Transform indices into distances from the origin. In general,
    it transforms rectangular shapes to circular ones.
    """
    return np.sqrt(indices[X] ** 2 + indices[Y] ** 2)


@overload
def shift_index_origin(
    indices: IdxMeshes,
    shifts: Sequence[int]
) -> IdxMeshes:
    ...


@overload
def shift_index_origin(
    indices: ImgAry,
    shifts: Sequence[int]
) -> ImgAry:
    ...


def shift_index_origin(indices, shifts):
    """Adjust the values in an indices array to move the origin point."""
    for axis, shift in enumerate(shifts):
        indices[axis] -= shift
    return indices


if __name__ == '__main__':
    from pjimg.util.debug import print_array

    def warp(a):
        return a + 0.25

    size = (1, 8, 8)
    source = Regular(5, 3, antialias=True)
    a = source.fill(size)
    a *= 0xff
    a = a.astype(np.uint8)
    print_array(a, depth=2)
