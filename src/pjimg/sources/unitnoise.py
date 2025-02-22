"""
Unit Noise
----------

Unit noise splits the image space into a grid of units, then randomly
distributes values at the positions on the grid. Image data is then
generated using an easing function that provides values for the pixels
between the positions on the grid.

.. autoclass:: pjimg.sources.UnitNoise
.. autoclass:: pjimg.sources.CosineNoise


Curtains
********
Curtains are unit noise that is generated in only one dimension of an image.

.. autoclass:: pjimg.sources.CosineCurtains
.. autoclass:: pjimg.sources.Curtains


Octave Noise
************
Octave noise creates more intricate patterns by adding together multiple
layers of unit noise.

.. autoclass:: pjimg.sources.OctaveUnitNoise
.. autoclass:: pjimg.sources.OctaveCosineCurtains
.. autoclass:: pjimg.sources.OctaveCurtains
.. autoclass:: pjimg.sources.BorktaveUnitNoise
.. autoclass:: pjimg.sources.BorktaveCosineNoise
.. autoclass:: pjimg.sources.BorktaveCosineCurtains

"""
from operator import mul, truediv
from typing import Callable, NamedTuple, Optional, Sequence, Union

import numpy as np

from pjimg.sources.model import Source
from pjimg.sources.noise import Noise, Seed
from pjimg.util import X, Y, Z, lerp
from pjimg.util.model import *


# Names available for import.
__all__ = [
    'BorktaveCosineCurtains', 'BorktaveCosineNoise', 'BorktaveUnitNoise',
    'CosineCurtains', 'CosineNoise', 'Curtains', 'OctaveCosineCurtains',
    'OctaveCurtains', 'OctaveUnitNoise', 'UnitNoise',
]


# Public classes.
class UnitNoise(Noise):
    """Create image noise that is based on a unit grid.
    
    :param unit: The number of pixels between vertices along an
        axis on the unit grid. The vertices are the locations where
        colors for the gradient are set. This is involved in setting
        the maximum size of noise that can be generated from
        the object.
    :param min: (Optional.) The minimum value of a vertex of the unit
        grid. This is involved in setting the maximum size of noise
        that can be generated from the object.
    :param max: (Optional.) The maximum value of a vertex of the unit
        grid. This is involved in setting the maximum size of noise
        that can be generated from the object.
    :param repeats: (Optional.) The number of times each value can
        appear on the unit grid. This is involved in setting the
        maximum size of noise that can be generated from the object.
    :param seed: (Optional.) An int, bytes, or string used to seed
        the random number generator used to generate the image data.
        If no value is passed, the RNG will not be seeded, so
        serialized versions of this source will not produce the
        same values. Note: strings that are passed to seed will
        be converted to UTF-8 bytes before being converted to
        integers for seeding.
    :param table: (Optional.) A table of values to use when generating
        the image data. If no value is passed, the table will be generated
        randomly. Default is `None`.
    :return: An instance of :class:`UnitNoise`.
    :rtype: sources.unitnoise.UnitNoise
    
    Usage::
    
        >>> # Create unit noise in a 1280x720 image.
        >>> size = (1, 720, 1280)
        >>> unit = (1, size[Y] // 5, size[Y] // 5)
        >>> source = UnitNoise(unit=unit, seed='spam')
        >>> img = source.fill(size)

    .. figure:: images/unitnoise.jpg
       :alt: Unit noise in a 1280x720 image.
       
       The image data created by the usage example.
    
    """
    # The number of dimensions the noise occurs in.
    _axes: int = 3

    def __init__(
        self, unit: Sequence[float],
        min: int = 0x00,
        max: int = 0xff,
        repeats: int = 0,
        seed: Seed = None,
        table: Optional[Sequence[int]] = None
    ) -> None:
        """Initialize an instance of UnitNoise."""
        # Initialize public values.
        self.unit = unit
        self.min = min
        self.max = max
        self.repeats = repeats
        super().__init__(seed)

        # Initialize the randomized table.
        if table is None:
            table = self._init_table()
        self._table = table

        # Prime the names of the grids used for interpolation.
        tmp = '{:>0' + str(self._axes) + 'b}'
        self._hashes = [tmp.format(n) for n in range(2 ** self._axes)]

    # Public methods.
    def fill(
        self, size: Size,
        location: Loc = (0, 0, 0)
    ) -> ImgAry:
        """Fill a volume with image data.

        :param size: The size of the volume of image data to generate.
        :param loc: (Optional.) How much to shift the starting point
            for the noise generation along each axis.
        :return: An :class:`numpy.ndarray` with image data.
        :rtype: numpy.ndarray
        """
        shape = self._calc_unit_grid_shape(size)
        whole, parts = self._map_unit_grid(size, location)
        grids = self._build_grids(whole, size, shape)
        a = self._interp(grids, parts)
        return a / self.max

    # Private methods.
    def _build_grids(
        self, whole: IntAry64,
        size: Size,
        shape: Size
    ) -> dict[str, IntAry64]:
        """Get the color for the eight vertices that surround each of
        the pixels.
        """
        grids = {}
        
        # The _hashes here are identifiers for each of the eight vertices
        # that surround any given point. It's a binary number with one
        # digit per dimension in the noise being generated. The digit
        # indicates whether to use the vertex before or after the given
        # point. Therefore, if the value of `key` is `010`, it would be
        # generating the grid based on the vertices before the point on
        # the Z and X axes but after the point on the Y axis.
        for key in self._hashes:
            
            # Identify the multidimensional indices of the vertex for
            # each point.
            grid_whole = whole.copy()
            a_grid = np.zeros(size, dtype=np.int64)
            for axis in range(self._axes):
                grid_whole[axis] += int(key[axis])

            # The values of the vertices are stored in a one-dimensional
            # array. This translates the multidimensional indices into
            # single-dimensional indices to allow a lookup in the vertices
            # value table,
            for axis in range(self._axes):
                remaining_axes = range(self._axes)[axis + 1:]
                axis_incr = 1
                for r_axis in remaining_axes:
                    axis_incr *= shape[r_axis]
                a_grid += grid_whole[axis] * axis_incr
                a_grid %= len(self._table)

            # Get the value of the vertex for each point and store it in
            # a dictionary by the identifier for the vertex.
            a_grid = np.take(self._table, a_grid)
            grids[key] = a_grid
        return grids

    def _calc_unit_grid_shape(self, size: Size):
        """Determine the shape of the unit grid."""
        shape = []
        for axis in range(self._axes):
            # Double inverse floor is ceiling division.
            length = -(-size[axis] // self.unit[axis])
            length = int(length)
            shape.append(length)

        return shape

    def _init_table(self) -> list[int]:
        """Create the table of randomized values for the unit grid."""
        table = []
        for repeat in range(self.repeats + 1):
            table.extend(list(range(self.min, self.max)))
        self._rng.shuffle(table)
        return table

    def _map_unit_grid(
        self, size: Sequence[int],
        location: Sequence[int]
    ) -> tuple[IntAry64, RatioAry]:
        """Map the image data to the unit grid."""
        # Map out the space.
        a = np.indices(size, float)
        for axis in range(self._axes):
            a[axis] += location[axis]

            # Split the space up into units.
            a[axis] = a[axis] / self.unit[axis]
            a[axis] %= 255

        # The unit distances are split. The unit values are needed
        # to set the color value of each vertex within the volume.
        # The parts value is needed to interpolate the noise value
        # at each pixel.
        whole = (a // 1).astype(int)
        parts = a - whole
        return whole, parts

    def _interp(
        self, grids: Union[
            dict[str, IntAry64],
            dict[str, RatioAry]
        ],
        parts: RatioAry
    ) -> ImgAry:
        """Interpolate the values of each pixel of image data."""
        if len(grids) > 2:
            new_grids = {}
            evens = [k for k in grids if k.endswith('0')]
            odds = [k for k in grids if k.endswith('1')]
            for even, odd in zip(evens, odds):
                new_key = even[:-1]
                axis = len(new_key)
                new_grids[new_key] = lerp(grids[even], grids[odd], parts[axis])
            return self._interp(new_grids, parts)

        return lerp(grids['0'], grids['1'], parts[Z])


class CosineNoise(UnitNoise):
    """Create image noise that is based on a unit grid and smoothed
    with a cosine ease.
    
    :param unit: The number of pixels between vertices along an
        axis on the unit grid. The vertices are the locations where
        colors for the gradient are set. This is involved in setting
        the maximum size of noise that can be generated from
        the object.
    :param min: (Optional.) The minimum value of a vertex of the unit
        grid. This is involved in setting the maximum size of noise
        that can be generated from the object.
    :param max: (Optional.) The maximum value of a vertex of the unit
        grid. This is involved in setting the maximum size of noise
        that can be generated from the object.
    :param repeats: (Optional.) The number of times each value can
        appear on the unit grid. This is involved in setting the
        maximum size of noise that can be generated from the object.
    :param seed: (Optional.) An int, bytes, or string used to seed
        the random number generator used to generate the image data.
        If no value is passed, the RNG will not be seeded, so
        serialized versions of this source will not produce the
        same values. Note: strings that are passed to seed will
        be converted to UTF-8 bytes before being converted to
        integers for seeding.
    :param table: (Optional.) A table of values to use when generating
        the image data. If no value is passed, the table will be generated
        randomly. Default is `None`.
    :return: An instance of :class:`CosineNoise`.
    :rtype: sources.unitnoise.CosineNoise
    
    Usage::
    
        >>> # Create unit noise in a 1280x720 image.
        >>> size = (1, 720, 1280)
        >>> unit = (1, size[Y] // 5, size[Y] // 5)
        >>> source = CosineNoise(unit=unit, seed='spam')
        >>> img = source.fill(size)

    .. figure:: images/cosinenoise.jpg
       :alt: Unit noise in a 1280x720 image.
       
       The image data created by the usage example.
    
    """
    # Private methods.
    def _map_unit_grid(
        self, size: Size,
        location: Loc
    ) -> tuple[IntAry64, RatioAry]:
        """Map the image data to the unit grid."""
        whole, parts = super()._map_unit_grid(size, location)
        parts = (1 - np.cos(parts * np.pi)) / 2
        return whole, parts


class Curtains(UnitNoise):
    """Unit noise that creates vertical lines, like curtains.
    
    :param unit: The number of pixels between vertices along an
        axis on the unit grid. The vertices are the locations where
        colors for the gradient are set. This is involved in setting
        the maximum size of noise that can be generated from
        the object.
    :param min: (Optional.) The minimum value of a vertex of the unit
        grid. This is involved in setting the maximum size of noise
        that can be generated from the object.
    :param max: (Optional.) The maximum value of a vertex of the unit
        grid. This is involved in setting the maximum size of noise
        that can be generated from the object.
    :param repeats: (Optional.) The number of times each value can
        appear on the unit grid. This is involved in setting the
        maximum size of noise that can be generated from the object.
    :param seed: (Optional.) An int, bytes, or string used to seed
        therandom number generator used to generate the image data.
        If no value is passed, the RNG will not be seeded, so
        serialized versions of this source will not produce the
        same values. Note: strings that are passed to seed will
        be converted to UTF-8 bytes before being converted to
        integers for seeding.
    :return: An instance of :class:`Curtains`.
    :rtype: sources.unitnoise.Curtains

    Usage::
    
        >>> # Create sharp curtains in a 1280x720 image.
        >>> size = (1, 720, 1280)
        >>> unit = (1, size[Y] // 5, size[Y] // 5)
        >>> source = Curtains(unit=unit, seed='spam')
        >>> img = source.fill(size)

    .. figure:: images/curtains.jpg
       :alt: Sharp curtains in a 1280x720 image.
       
       The image data created by the usage example.

    """
    # The number of dimensions the noise occurs in.
    _axes: int = 2

    # Public methods.
    def fill(
        self, size: Size,
        loc: Loc = (0, 0, 0)
    ) -> ImgAry:
        """Return a space filled with noise."""
        noise_size = (size[Z], size[X])
        noise_loc = (loc[Z], loc[X])
        a = super().fill(noise_size, noise_loc)
        return np.tile(a[:, np.newaxis, ...], (1, size[Y], 1))


class CosineCurtains(Curtains):
    """Unit noise that creates vertical lines with a cosine-based ease
    on the color change between grid points, making them appear to
    flow more like curtains.
    
    :param unit: The number of pixels between vertices along an
        axis on the unit grid. The vertices are the locations where
        colors for the gradient are set. This is involved in setting
        the maximum size of noise that can be generated from
        the object.
    :param min: (Optional.) The minimum value of a vertex of the unit
        grid. This is involved in setting the maximum size of noise
        that can be generated from the object.
    :param max: (Optional.) The maximum value of a vertex of the unit
        grid. This is involved in setting the maximum size of noise
        that can be generated from the object.
    :param repeats: (Optional.) The number of times each value can
        appear on the unit grid. This is involved in setting the
        maximum size of noise that can be generated from the object.
    :param seed: (Optional.) An int, bytes, or string used to seed
        therandom number generator used to generate the image data.
        If no value is passed, the RNG will not be seeded, so
        serialized versions of this source will not produce the
        same values. Note: strings that are passed to seed will
        be converted to UTF-8 bytes before being converted to
        integers for seeding.
    :return: An instance of :class:`CosineCurtains`.
    :rtype: sources.unitnoise.CosineCurtains
    
    Usage::
    
        >>> # Create rounded curtains in a 1280x720 image.
        >>> size = (1, 720, 1280)
        >>> unit = (1, size[Y] // 5, size[Y] // 5)
        >>> source = CosineCurtains(unit=unit, seed='spam')
        >>> img = source.fill(size)

    .. figure:: images/cosinecurtains.jpg
       :alt: Rounded curtains in a 1280x720 image.
       
       The image data created by the usage example.

    """
    # Private methods.
    def _map_unit_grid(
        self, size: Size,
        location: Loc
    ) -> tuple[IntAry64, RatioAry]:
        """Map the image data to the unit grid."""
        whole, parts = super()._map_unit_grid(size, location)
        parts = (1 - np.cos(parts * np.pi)) / 2
        return whole, parts


# Factories.
class OctaveNoiseDefaults(NamedTuple):
    """The default values for a new octave noise class created by
    :func:`sources.unitnoise.octave_noise_factory`.
    
    :param octaves: (Optional.) The default value of octaves.
    :param persistence: (Optional.) The default value of persistence.
    :param amplitude: (Optional.) The default value of amplitude.
    :param frequency: (Optional.) The default value of frequency.
    :param unit: (Optional.) The default value of unit.
    :param min: (Optional.) The default value of min.
    :param max: (Optional.) The default value of max.
    :param repeats: (Optional.) The default value of repeats.
    :param seed: (Optional.) The default value of seed.
    :return: An instance of :class:`OctaveNoiseDefaults`.
    :rtyoe: sources.unitnoise.OctaveNoiseDefaults
    """
    octaves: int = 4
    persistence: float = 8
    amplitude: float = 8
    frequency: float = 2
    unit: Sequence[int] = (1024, 1024, 1024)
    min: int = 0x00
    max: int = 0xff
    repeats: int = 1
    seed: Seed = None


def octave_noise_factory(
    source: type[UnitNoise],
    defaults: OctaveNoiseDefaults,
    bork: bool = False
) -> type:
    """A class factory that generates octave versions of the subclasses
    of :class:`UnitNoise`.
    
    Octave noise contains multiple layers, called "octaves," of the
    noise being generated and added together. The size of the units
    within each octave changes based on the frequency. The amount
    each octave affects the final image is based on the amplitude
    and persistence.
    
    .. warning:
        This factory only works with subclasses of :class:`UnitNoise`.
        This is because the octave algorithm uses the unit size of the
        source to affect the size of the variations within the noise.
        Other types of noise may be able to be octaved, but they will
        need a different algorithm than the one used by this factory.
    
    :param source: The type of :class:`UnitNoise` to use when generating
        the octave noise.
    :param defaults: The default values for the parameters of the class
        being created.
    :param bork: This changes the operator used to calculate the new
        unit sizes each octave from division to multiplication. This
        was a bug in earlier versions of this code that had some
        interesting effects. So, while the behavior is fixed by default,
        this allows you to revert to the older, broken behavior.
    :return: A subclass of :class:`UnitNoise`.
    :rtype: type
    """
    class OctaveNoise(Source):
        """A source for octave noise. Parameters are similar to the
        :class:`UnitNoise` being octaved, with the following additions.
        
        :param octaves: The number of octaves of noise in the image. An
            octave is a layer of the noise with a different number of
            points added on top of other layers of noise.
        :param persistence: How the weight of each octave changes.
        :param amplitude: The weight of the first octave.
        :param frequency: How the number of points in each octave changes.
        :return: An :class:`pjimg.sources.{name}` object.
        :rtype: pjimg.sources.{name}
        
        Usage::
        
        >>> # Create {name} noise in a 1280x720 image.
        >>> size = (1, 720, 1280)
        >>> unit = (1, size[Y] // 5, size[Y] // 5)
        >>> source = {name}(
        >>>     unit=unit,
        >>>     octaves={octaves},
        >>>     persistence={persistence},
        >>>     amplitude={amplitude},
        >>>     frequency={frequency},
        >>>     seed='spam'
        >>> )
        >>> img = source.fill(size)

        .. figure:: images/{lname}.jpg
           :alt: A picture of an image created from the output of
                :class:`{name}`.
       
           Output of :class:`{name}`.
    
        """
        source: type[UnitNoise]
        unit_op: Callable[[float, float], float] = truediv

        def __init__(
            self, octaves: int = defaults.octaves,
            persistence: float = defaults.persistence,
            amplitude: float = defaults.amplitude,
            frequency: float = defaults.frequency,
            unit: Sequence[float] = defaults.unit,
            min: int = defaults.min,
            max: int = defaults.max,
            repeats: int = defaults.repeats,
            seed: Seed = defaults.seed,
            table: Optional[Sequence[int]] = None
        ) -> None:
            self.octaves = octaves
            self.persistence = persistence
            self.amplitude = amplitude
            self.frequency = frequency
            self.unit = unit
            self.min = min
            self.max = max
            self.repeats = repeats
            self.seed = seed
            self.table = table

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
            a = np.zeros(tuple(size), dtype=float)
            max_value = 0.0
            for i in range(self.octaves):
                amp = self.amplitude + (self.persistence * i)
                freq = self.frequency * 2 ** i
                unit = [self.unit_op(n, freq) for n in self.unit]
                octave = self.source(
                    unit=unit,
                    min=self.min,
                    max=self.max,
                    repeats=self.repeats,
                    seed=self.seed,
                    table=self.table
                )
                a += octave.fill(size, loc) * amp
                max_value += amp
            a /= max_value
            return a

    cls = OctaveNoise
    cls.source = source
    if not bork:
        cls.__name__ = 'Octave' + source.__name__
        if cls.__doc__ is not None:
            cls.__doc__ = cls.__doc__.format(
                lname='octave' + source.__name__.lower(),
                name='Octave' + source.__name__,
                octaves=3,
                persistence=-4,
                amplitude=24,
                frequency=4
            )
    else:
        cls.unit_op = mul
        cls.__name__ = 'Borktave' + source.__name__
        if cls.__doc__ is not None:
            cls.__doc__ = cls.__doc__.format(
                lname='borktave' + source.__name__.lower(),
                name='Borktave' + source.__name__,
                octaves=4,
                persistence=8,
                amplitude=8,
                frequency=-1.5
            )
            cls.__doc__ += '\n'.join((
                '',
                '.. warning::',
                '   This octave class is borked. That means the operation',
                '   used to calculate the unit size for each octave',
                '   multiplies by the frequency rather than divides. This',
                '   reproduces a bug in earlier versions of this code which',
                '   caused some interesting behavior. It generates output',
                '   just fine. It just won\'t be the same output generated',
                '   with the real octave algorithm.'
            ))
    return cls


# Octave unit noise classes.
defaults = OctaveNoiseDefaults()
OctaveCosineCurtains = octave_noise_factory(CosineCurtains, defaults)
OctaveCurtains = octave_noise_factory(Curtains, defaults)
OctaveUnitNoise = octave_noise_factory(UnitNoise, defaults)
BorktaveCosineCurtains = octave_noise_factory(CosineCurtains, defaults, True)
BorktaveCosineNoise = octave_noise_factory(CosineNoise, defaults, True)
BorktaveUnitNoise = octave_noise_factory(UnitNoise, defaults, True)
