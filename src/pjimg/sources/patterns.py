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
.. autoclass:: pjimg.sources.Rings
.. autoclass:: pjimg.sources.Solid
.. autoclass:: pjimg.sources.Spheres
.. autoclass:: pjimg.sources.Spot
.. autoclass:: pjimg.sources.Text
.. autoclass:: pjimg.sources.Waves

"""
from math import sqrt
from typing import Callable, Literal, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pjimg.sources.model import Source
from pjimg.util import ImgAry, Loc, Size, X, Y, Z


# Names available for import.
__all__ = [
    'Box', 'Gradient', 'Hexes', 'Lines', 'Radials', 'Rays', 'Rings',
    'Solid', 'Spheres', 'Spot', 'Text', 'Waves',
]


# Public classes.
class Box(Source):
    """Draw a box.
    
    :param origin: The location of the upper left corner of the box.
    :param dimensions: The size of the box in three dimensions.
    :param color: The color of the box. This is a float within the
        range 0 <= x <= 1.
    :return: A :class:`Box` object.
    :rtype: sources.patterns.Box

    Usage::
    
        >>> # Create an image of a gray rectangle in the middle of a
        >>> # 1280x720 image.
        >>> size = (1, 720, 1280)
        >>> origin = (n // 4 for n in size)
        >>> dimensions = (1, *(n // 2 for n in size[Y:]))
        >>> source = Box(origin=origin, dimensions=dimensions, color=0.5)
        >>> img = source.fill(size)
    
    .. figure:: images/box.jpg
       :alt: An image of a gray rectangle in the middle of a 1280x720 image.
       
       The image data created by the usage example.

    """
    def __init__(
        self, origin: Loc,
        dimensions: Sequence[int],
        color: float = 1.0
    ) -> None:
        self.origin = origin
        self.dimensions = dimensions
        self.color = color

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
        a = np.zeros(size)
        start = [n + o for n, o in zip(loc, self.origin)]
        end = [s + d for s, d in zip(start, self.dimensions)]
        slices = [slice(s, e) for s, e in zip(start, end)]
        a[tuple(slices)] = self.color
        return a


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

    Usage::
    
        >>> # Create a horizontal gradient with multiple stops in a
        >>> # 1280x720 image.
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
       :alt: A horizontal gradient with multiple stops in a 1280x720 image.
       
       The image data created by the usage example.

    """
    def __init__(
        self, direction: str = 'h',
        stops: Sequence[float] = (0, 0, 1, 1)
    ) -> None:
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

    Usage::
    
        >>> # Create a hexagonal grid of cells in a 1280x720 image.
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
        round: bool = False
    ) -> None:
        self.cells = cells
        self.radius = radius
        self.round = round
    
    # Public methods.
    def fill(
        self, size: Sequence[int],
        loc: Sequence[int] = (0, 0, 0)
    ) -> ImgAry:
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


class Lines(Source):
    """Generate simple lines.

    :param direction: (Optional.) This should be 'h' for a horizontal
        gradient or 'v' for a vertical gradient.
    :param length: (Optional.) The distance between each line.
    :return: :class:`Lines` object.
    :rtype: sources.patterns.Lines

    Usage::
    
        >>> # Create a series of vertical lines in a 1280x720 image.
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
        length: float = 64
    ) -> None:
        self.direction = direction
        self.length = float(length)

    # Public methods.
    def fill(
        self, size: Sequence[int],
        loc: Sequence[int] = (0, 0, 0)
    ) -> ImgAry:
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
        return values


class Radials(Source):
    """Generates concentric radial gradients.

    :param length: The radius of the innermost circle.
    :param growth: (Optional.) Either the string 'linear' or the
        string 'geometric'. Determines whether the distance between
        each circle remains constant (linear) or increases
        (geometric). Defaults to linear.
    :returns: :class:`Radials` object.
    :rtype: sources.patterns.Radials

    Usage::
    
        >>> # Create a series of concentric radial gradients in a
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
        growth: str = 'l'
    ) -> None:
        """Initialize an instance of Waves."""
        self.length = float(length)
        self.growth = growth

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
        offset: float = 0
    ) -> None:
        self.count = int(count)
        self.offset = float(offset)

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
    :return: :class:`Rings` object.
    :rtype: sources.patterns.Rings

    Usage::
    
        >>> # Create a series of concentric rings in a 1280x720 image.
        >>> size = (1, 720, 1280)
        >>> radius = size[X] / 6
        >>> width = size[X] / 12
        >>> gap = size[X] / 18
        >>> count = 6
        >>> source = Rings(radius=radius, width=width, gap=gap, count=count)
        >>> img = source.fill(size)
    
    .. figure:: images/rings.jpg
       :alt: A series of concentric rings in a 1280.720 image.
       
       The image data created by the usage example.

    """
    def __init__(
        self, radius: float,
        width: float,
        gap: float = 0,
        count: int = 1
    ) -> None:
        """Initialize an instance of Ring."""
        self.radius = float(radius)
        self.width = float(width)
        self.gap = float(gap)
        self.count = int(count)

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
        # Map out the volume of space that will be created.
        a = np.zeros(size, dtype=float)
        c = np.indices(size, dtype=float)
        c = center_index_origin(c)
        c = shift_index_origin(c, loc)

        # Perform a spherical interpolation on the points in the
        # volume and run the easing function on the results.
        c = index_to_distance_from_origin(c)
        for i in range(self.count):
            radius = self.radius + self.gap * i
            if radius != 0:
                working = c / np.sqrt(radius ** 2)
                working = np.abs(working - 1)
                wr = self.width / 2 / radius
                m = np.zeros(working.shape, bool)
                m[working <= wr] = True
                a[m] = working[m] * (radius / (self.width / 2))
                a[m] = 1 - a[m]
        return a


class Solid(Source):
    """Fill a space with a solid color.

    :param color: The color to use for the fill. Zero is black. One
        is white. The values between are values of gray.
    :return: :class:`Solid` object.
    :rtype: pjinoise.sources.Solid

    Usage::
    
        >>> # Create a solid gray 1280x720 image.
        >>> size = (1, 1280, 720)
        >>> source = Solid(color=0.25)
        >>> img = source.fill(size)
    
    .. figure:: images/solid.jpg
       :alt: A solid gray 1280x720 image.
       
       The image data created by the usage example.

    """
    def __init__(self, color: float) -> None:
        self.color = float(color)

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
        a = np.zeros(size, dtype=float)
        a.fill(self.color)
        return a


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

    Usage::
    
        >>> # Create a square grid of cells in a 1280x720 image.
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
        round: bool = True
    ) -> None:
        self.cells = cells
        self.offset = offset
        self.radius = float(radius)
        self.round = round

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


class Spot(Source):
    """Fill a space with a spot.

    :param radius: The radius of the spot.
    :return: :class:`Spot` object.
    :rtype: sources.patterns.Spot
    
    Usage::
    
        >>> # Create a radial gradient centered in a 1280x720 image.
        >>> size = (1, 720, 1280)
        >>> radius = size[Y] * 2 / 3
        >>> source = Spot(radius=radius)
        >>> img = source.fill(size)

    .. figure:: images/spot.jpg
       :alt: A radial gradient centered in a 1280x720 image.
       
       The image data created by the usage example.

    """
    def __init__(self, radius: float, *args, **kwargs) -> None:
        self.radius = float(radius)

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
    
    Usage::
    
        >>> # Create the word "SPAM" in a 1280x720 image.
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
        stroke_fill: int = 0
    ) -> None:
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
    
    Usage::
    
        >>> # Create a wave pattern in a 1280x720 image.
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
        warp: Optional[Callable[[ImgAry], ImgAry]] = None,
        radial: bool = False
    ) -> None:
        self.unit = unit
        self.angle = angle
        self.wavelength = wavelength
        self.warp = warp
        self.radial = radial
    
    def fill(self, size: Size, loc: Loc = (0, 0, 0)) -> ImgAry:
        x = self.angle / 90
        indices = np.indices(size, dtype=float)
        f = (2 * np.pi) / self.wavelength
        
        # Set the angle of the wave.
        if not self.radial:
            a = indices[X] * (1 - x) + indices[Y] * x
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


def shift_index_origin(indices: ImgAry, shifts: Sequence[int]) -> ImgAry:
    """Adjust the values in an indices array to move the origin point."""
    for axis, shift in enumerate(shifts):
        indices[axis] -= shift
    return indices


if __name__ == '__main__':
    from pjimg.util.debug import print_array
    
    def warp(a):
        return a + 0.25

    size = (1, 8, 8)
    source = Waves(unit=7, radial=True)
    a = source.fill(size)
    a *= 0xff
    a = a.astype(np.uint8)
    print_array(a)
