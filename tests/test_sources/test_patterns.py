"""
test_patterns
~~~~~~~~~~~~~

Unit tests for :mod:`pjimg.sources.patterns`.
"""
import numpy as np

from pjimg.sources import patterns as p
from tests.common import mkhex


# Test cases.
class TestBox:
    # Tests for initialization.
    def test_init_all_default(self):
        """Given only required parameters, :class:`Box` should
        initialize the required attributes with the given values. It
        should then initialize the optional attributes with default
        values.
        """
        required = {
            'origin': (4, 4, 4),
            'dimensions': (4, 4),
        }
        optional = {
            'color': 1.0,
        }
        obj = p.Box(**required)
        for attr in required:
            assert getattr(obj, attr) == required[attr]
        for attr in optional:
            assert getattr(obj, attr) == optional[attr]

    def test_init_all_optional(self):
        """Given optional parameters, :class:`Box` should
        initialize the given attributes with the given values.
        """
        required = {
            'origin': (4, 4, 4),
            'dimensions': (4, 4),
        }
        optional = {
            'color': 0.5,
        }
        obj = p.Box(**required, **optional)
        for attr in required:
            assert getattr(obj, attr) == required[attr]
        for attr in optional:
            assert getattr(obj, attr) == optional[attr]

    # Tests for fill.
    def test_fill(self):
        """Given origin, dimensions, and a color, :meth:`Box.fill`
        should return a volume filled with a box of the origin,
        dimensions, and color given when the object was created.
        """
        obj = p.Box((0, 1, 1), (1, 2, 3), 0x80 / 0xff)
        result = obj.fill((2, 8, 8))
        assert (mkhex(result) == np.array([
            [
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x80, 0x80, 0x80, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x80, 0x80, 0x80, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            ],
            [
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            ],
        ], dtype=np.uint8)).all()


class TestGradient:
    # Tests for initialization.
    def test_init_all_default(self):
        """Given no parameters, :class:`Gradient` should
        initialize the required attributes with the given values. It
        should then initialize the optional attributes with default
        values.
        """
        optional = {
            'direction': 'h',
            'stops': (0, 0, 1, 1),
        }
        obj = p.Gradient()
        optional['stops'] = [[0, 0], [1, 1]]
        for attr in optional:
            assert getattr(obj, attr) == optional[attr]

    def test_init_all_optional(self):
        """Given optional parameters, :class:`Gradient` should
        initialize the given attributes with the given values.
        """
        optional = {
            'direction': 'v',
            'stops': (0.2, 0.4, 0.6, 0.8),
        }
        obj = p.Gradient(**optional)
        optional['stops'] = [[0, 0.4], [0.2, 0.4], [0.6, 0.8], [1, 0.8]]
        for attr in optional:
            assert getattr(obj, attr) == optional[attr]

    # Tests for fill.
    def test_fill(self):
        """Given origin, dimensions, and a color, :meth:`Gradient.fill`
        should return a volume filled with a box of the origin,
        dimensions, and color given when the object was created.
        """
        obj = p.Gradient(direction='v', stops=[0, 0, .5, 1, 1, 0])
        result = obj.fill((2, 5, 4))
        assert (mkhex(result) == np.array([
            [
                [0x00, 0x00, 0x00, 0x00],
                [0x7f, 0x7f, 0x7f, 0x7f],
                [0xff, 0xff, 0xff, 0xff],
                [0x7f, 0x7f, 0x7f, 0x7f],
                [0x00, 0x00, 0x00, 0x00],
            ],
            [
                [0x00, 0x00, 0x00, 0x00],
                [0x7f, 0x7f, 0x7f, 0x7f],
                [0xff, 0xff, 0xff, 0xff],
                [0x7f, 0x7f, 0x7f, 0x7f],
                [0x00, 0x00, 0x00, 0x00],
            ],
        ], dtype=np.uint8)).all()


class TestHexes:
    # Tests for initialization.
    def test_init_all_default(self):
        """Given only required parameters, :class:`Hexes` should
        initialize the required attributes with the given values.
        It should then initialize the optional attributes with
        default values.
        """
        required = {
            'radius': 2.0,
        }
        optional = {
            'cells': True,
            'round': False,
        }
        obj = p.Hexes(**required)
        for attr in required:
            assert getattr(obj, attr) == required[attr]
        for attr in optional:
            assert getattr(obj, attr) == optional[attr]

    def test_init_all_optional(self):
        """Given optional parameters, :class:`Hexes` should
        initialize the given attributes with the given values.
        """
        required = {
            'radius': 2.0,
        }
        optional = {
            'cells': False,
            'round': True,
        }
        obj = p.Hexes(**required, **optional)
        for attr in required:
            assert getattr(obj, attr) == required[attr]
        for attr in optional:
            assert getattr(obj, attr) == optional[attr]

    # Tests for fill.
    def test_fill(self):
        """Given the shape of the output, :meth:`Hexes.fill`
        should return a volume filled with hexagons of the
        given radius.
        """
        obj = p.Hexes(radius=5)
        result = obj.fill((1, 8, 8))
        assert (mkhex(result) == np.array([
            [
                [0xff, 0xa4, 0x4a, 0x4a, 0xa4, 0xff, 0xa4, 0x4a],
                [0xa4, 0x7f, 0x35, 0x35, 0x7f, 0xa4, 0x7f, 0x35],
                [0x4a, 0x35, 0x28, 0x28, 0x35, 0x4a, 0x35, 0x28],
                [0x00, 0x4a, 0x7f, 0x7f, 0x4a, 0x00, 0x4a, 0x7f],
                [0x1b, 0x74, 0xc9, 0xc9, 0x74, 0x1b, 0x74, 0xc9],
                [0x15, 0x6b, 0xb3, 0xb3, 0x6b, 0x15, 0x6b, 0xb3],
                [0x0f, 0x34, 0x62, 0x62, 0x34, 0x0f, 0x34, 0x62],
                [0x69, 0x50, 0x14, 0x14, 0x50, 0x69, 0x50, 0x14],
            ],
        ], dtype=np.uint8)).all()


class TestLines:
    # Tests for initialization.
    def test_init_all_default(self):
        """Given no parameters, :class:`Lines` should initialize
        the required attributes with the given values. It should
        then initialize the optional attributes with default values.
        """
        optional = {
            'direction': 'h',
            'length': 64,
        }
        obj = p.Lines()
        for attr in optional:
            assert getattr(obj, attr) == optional[attr]

    def test_init_all_optional(self):
        """Given optional parameters, :class:`Lines` should
        initialize the given attributes with the given values.
        """
        optional = {
            'direction': 'h',
            'length': 64,
        }
        obj = p.Lines(**optional)
        for attr in optional:
            assert getattr(obj, attr) == optional[attr]

    # Tests for fill.
    def test_fill(self):
        """Given origin, dimensions, and a color, :meth:`Lines.fill`
        should return a volume filled with a box of the origin,
        dimensions, and color given when the object was created.
        """
        obj = p.Lines(direction='h', length=5)
        result = obj.fill((2, 4, 4))
        assert (mkhex(result) == np.array([
            [
                [0x00, 0x00, 0x00, 0x00],
                [0x7f, 0x7f, 0x7f, 0x7f],
                [0xff, 0xff, 0xff, 0xff],
                [0x7f, 0x7f, 0x7f, 0x7f],
            ],
            [
                [0x7f, 0x7f, 0x7f, 0x7f],
                [0xff, 0xff, 0xff, 0xff],
                [0x7f, 0x7f, 0x7f, 0x7f],
                [0x00, 0x00, 0x00, 0x00],
            ],
        ], dtype=np.uint8)).all()


class TestRadials:
    # Tests for initialization.
    def test_init_all_default(self):
        """Given only required parameters, :class:`Radials` should
        initialize the required attributes with the given values.
        It should then initialize the optional attributes with
        default values.
        """
        required = {
            'length': 2.0,
        }
        optional = {
            'growth': 'l',
        }
        obj = p.Radials(**required)
        for attr in required:
            assert getattr(obj, attr) == required[attr]
        for attr in optional:
            assert getattr(obj, attr) == optional[attr]

    def test_init_all_optional(self):
        """Given optional parameters, :class:`Radials` should
        initialize the given attributes with the given values.
        """
        required = {
            'length': 2.0,
        }
        optional = {
            'growth': 'g',
        }
        obj = p.Radials(**required, **optional)
        for attr in required:
            assert getattr(obj, attr) == required[attr]
        for attr in optional:
            assert getattr(obj, attr) == optional[attr]

    # Tests for fill.
    def test_fill(self):
        """Given the shape of an output array, :meth:`Radials.fill`
        should return a volume filled with a box of the origin,
        dimensions, and color given when the object was created.
        """
        obj = p.Radials(length=3, growth='l')
        result = obj.fill((1, 8, 8))
        assert (mkhex(result) == np.array([
            [
                [0x4c, 0x21, 0x75, 0xa3, 0xa3, 0x75, 0x21, 0x4c],
                [0x21, 0xa3, 0xf0, 0xb2, 0xb2, 0xf0, 0xa3, 0x21],
                [0x75, 0xf0, 0x69, 0x0d, 0x0d, 0x69, 0xf0, 0x75],
                [0xa3, 0xb2, 0x0d, 0x86, 0x86, 0x0d, 0xb2, 0xa3],
                [0xa3, 0xb2, 0x0d, 0x86, 0x86, 0x0d, 0xb2, 0xa3],
                [0x75, 0xf0, 0x69, 0x0d, 0x0d, 0x69, 0xf0, 0x75],
                [0x21, 0xa3, 0xf0, 0xb2, 0xb2, 0xf0, 0xa3, 0x21],
                [0x4c, 0x21, 0x75, 0xa3, 0xa3, 0x75, 0x21, 0x4c],
            ],
        ], dtype=np.uint8)).all()


class TestRays:
    # Tests for initialization.
    def test_init_all_default(self):
        """Given only required parameters, :class:`Rays` should
        initialize the required attributes with the given values.
        It should then initialize the optional attributes with
        default values.
        """
        required = {
            'count': 2,
        }
        optional = {
            'offset': 0.0
        }
        obj = p.Rays(**required)
        for attr in required:
            assert getattr(obj, attr) == required[attr]
        for attr in optional:
            assert getattr(obj, attr) == optional[attr]

    def test_init_all_optional(self):
        """Given optional parameters, :class:`Rays` should
        initialize the given attributes with the given values.
        """
        required = {
            'count': 3,
        }
        optional = {
            'offset': 0.5
        }
        obj = p.Rays(**required, **optional)
        for attr in required:
            assert getattr(obj, attr) == required[attr]
        for attr in optional:
            assert getattr(obj, attr) == optional[attr]

    # Tests for fill.
    def test_fill(self):
        """Given origin, dimensions, and a color, :meth:`Rays.fill`
        should return a volume filled with a rays of the count,
        and rotation given when the object was initialized.
        """
        obj = p.Rays(count=3, offset=np.pi / 2)
        result = obj.fill((1, 8, 8))
        assert (mkhex(result) == np.array([
            [
                [0x89, 0x60, 0x2c, 0x13, 0x58, 0x98, 0xcd, 0xf5],
                [0xb1, 0x89, 0x4d, 0x06, 0x66, 0xb9, 0xf5, 0xe0],
                [0xe5, 0xc4, 0x89, 0x18, 0x84, 0xf5, 0xcc, 0xab],
                [0xd8, 0xe5, 0xf9, 0x89, 0xf5, 0x97, 0x79, 0x6b],
                [0x93, 0x85, 0x67, 0x09, 0x75, 0x05, 0x19, 0x26],
                [0x53, 0x32, 0x09, 0x7a, 0xe6, 0x75, 0x3a, 0x19],
                [0x1e, 0x09, 0x45, 0x98, 0xf8, 0xb1, 0x75, 0x4d],
                [0x09, 0x31, 0x66, 0xa6, 0xeb, 0xd2, 0x9e, 0x75],
            ],
        ], dtype=np.uint8)).all()


class TestRegular:
    # Tests for initialization.
    def test_init_all_default(self):
        """Given only required parameters, :class:`Regular` should
        initialize the required attributes with the given values.
        It should then initialize the optional attributes with
        default values.
        """
        required = {
            'sides': 3,
            'rho': 3.0,
        }
        optional = {
            'rotate': 0.0,
            'color': 1.0,
            'bg_color': 0.0,
            'antialias': False,
        }
        obj = p.Regular(**required)
        for attr in required:
            assert getattr(obj, attr) == required[attr]
        for attr in optional:
            assert getattr(obj, attr) == optional[attr]

    def test_init_all_optional(self):
        """Given optional parameters, :class:`Regular` should
        initialize the given attributes with the given values.
        """
        required = {
            'sides': 3,
            'rho': 3.0,
        }
        optional = {
            'rotate': 0.1,
            'color': 0.5,
            'bg_color': 1.0,
            'antialias': True,
        }
        obj = p.Regular(**required, **optional)
        for attr in required:
            assert getattr(obj, attr) == required[attr]
        for attr in optional:
            assert getattr(obj, attr) == optional[attr]

    # Tests for fill.
    def test_fill(self):
        """Given origin, dimensions, and a color, :meth:`Regular.fill`
        should return a volume filled with a regular polygon of the
        number of sides and distance from center to vertex (rho) given
        when the object was created.
        """
        obj = p.Regular(5, 3)
        result = obj.fill((1, 8, 8))
        assert (mkhex(result) == np.array([
            [
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0xff, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00],
                [0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            ],
        ], dtype=np.uint8)).all()

    def test_fill_antialias(self):
        """Given origin, dimensions, and a color, :meth:`Regular.fill`
        should return a volume filled with a regular polygon of the
        number of sides and distance from center to vertex (rho) given
        when the object was created. If antialiased, the edge of the
        polygon should be antialiased.
        """
        obj = p.Regular(5, 3, antialias=True)
        result = obj.fill((1, 8, 8))
        assert (mkhex(result) == np.array([
            [
                [0x00, 0x00, 0x00, 0x1f, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x25, 0xff, 0xaa, 0x12, 0x00, 0x00],
                [0x00, 0x48, 0xff, 0xff, 0xff, 0xe7, 0x25, 0x00],
                [0x3d, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x21],
                [0x14, 0xd6, 0xff, 0xff, 0xff, 0xff, 0xdb, 0x14],
                [0x00, 0x8c, 0xff, 0xff, 0xff, 0xff, 0x8c, 0x00],
                [0x00, 0x21, 0xff, 0xff, 0xff, 0xff, 0x23, 0x00],
                [0x00, 0x00, 0x1f, 0x3a, 0x3a, 0x21, 0x00, 0x00],
            ],
        ], dtype=np.uint8)).all()

    def test_fill_bg_color(self):
        """Given origin, dimensions, and a color, :meth:`Regular.fill`
        should return a volume filled with a regular polygon of the
        number of sides and distance from center to vertex (rho) given
        when the object was created. If background color is given, the
        area outside the polygon should be filled with the given color.
        """
        obj = p.Regular(5, 3, bg_color=0.5)
        result = obj.fill((1, 8, 8))
        assert (mkhex(result) == np.array([
            [
                [0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f],
                [0x7f, 0x7f, 0x7f, 0xff, 0x7f, 0x7f, 0x7f, 0x7f],
                [0x7f, 0x7f, 0xff, 0xff, 0xff, 0xff, 0x7f, 0x7f],
                [0x7f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f],
                [0x7f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f],
                [0x7f, 0x7f, 0xff, 0xff, 0xff, 0xff, 0x7f, 0x7f],
                [0x7f, 0x7f, 0xff, 0xff, 0xff, 0xff, 0x7f, 0x7f],
                [0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f],
            ],
        ], dtype=np.uint8)).all()

    def test_fill_color(self):
        """Given origin, dimensions, and a color, :meth:`Regular.fill`
        should return a volume filled with a regular polygon of the
        number of sides and distance from center to vertex (rho) given
        when the object was created. If color is given, the polygon
        should be filled with the given color.
        """
        obj = p.Regular(5, 3, color=0.5)
        result = obj.fill((1, 8, 8))
        assert (mkhex(result) == np.array([
            [
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x7f, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x7f, 0x7f, 0x7f, 0x7f, 0x00, 0x00],
                [0x00, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x00],
                [0x00, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x00],
                [0x00, 0x00, 0x7f, 0x7f, 0x7f, 0x7f, 0x00, 0x00],
                [0x00, 0x00, 0x7f, 0x7f, 0x7f, 0x7f, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            ],
        ], dtype=np.uint8)).all()

    def test_fill_rotate(self):
        """Given origin, dimensions, and a color, :meth:`Regular.fill`
        should return a volume filled with a regular polygon of the
        number of sides and distance from center to vertex (rho) given
        when the object was created. When given a value of rotate in
        radians, the polygon should be rotated clockwise by that amount.
        """
        rotate = 2 * np.pi / 10
        obj = p.Regular(5, 3, rotate=rotate)
        result = obj.fill((1, 8, 8))
        assert (mkhex(result) == np.array([
            [
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00],
                [0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0xff, 0x00, 0x00, 0x00],
            ],
        ], dtype=np.uint8)).all()


class TestRing:
    # Tests for initialization.
    def test_init_all_default(self):
        """Given only required parameters, :class:`Rings` should
        initialize the required attributes with the given values.
        It should then initialize the optional attributes with
        default values.
        """
        required = {
            'radius': 2.0,
            'width': 1.5,
        }
        optional = {
            'gap': 0.0,
            'count': 1,
        }
        obj = p.Rings(**required)
        for attr in required:
            assert getattr(obj, attr) == required[attr]
        for attr in optional:
            assert getattr(obj, attr) == optional[attr]

    def test_init_all_optional(self):
        """Given optional parameters, :class:`Rings` should
        initialize the given attributes with the given values.
        """
        required = {
            'radius': 2.0,
            'width': 1.5,
        }
        optional = {
            'gap': 0.5,
            'count': 2,
        }
        obj = p.Rings(**required, **optional)
        for attr in required:
            assert getattr(obj, attr) == required[attr]
        for attr in optional:
            assert getattr(obj, attr) == optional[attr]

    # Tests for fill.
    def test_fill(self):
        """Given origin, dimensions, and a color, :meth:`Rings.fill`
        should return a volume filled with a box of the origin,
        dimensions, and color given when the object was created.
        """
        obj = p.Rings(radius=2, width=1, gap=2, count=3)
        result = obj.fill((1, 8, 8))
        assert (mkhex(result) == np.array([
            [
                [0x00, 0x65, 0x9d, 0x12, 0x12, 0x9d, 0x65, 0x00],
                [0x65, 0x12, 0x00, 0x00, 0x00, 0x00, 0x12, 0x65],
                [0x9d, 0x00, 0xc1, 0x29, 0x29, 0xc1, 0x00, 0x9d],
                [0x12, 0x00, 0x29, 0x00, 0x00, 0x29, 0x00, 0x12],
                [0x12, 0x00, 0x29, 0x00, 0x00, 0x29, 0x00, 0x12],
                [0x9d, 0x00, 0xc1, 0x29, 0x29, 0xc1, 0x00, 0x9d],
                [0x65, 0x12, 0x00, 0x00, 0x00, 0x00, 0x12, 0x65],
                [0x00, 0x65, 0x9d, 0x12, 0x12, 0x9d, 0x65, 0x00],
            ],
        ], dtype=np.uint8)).all()


class TestSolid:
    # Tests for initialization.
    def test_init_all_default(self):
        """Given only required parameters, :class:`Solid` should
        initialize the required attributes with the given values.
        It should then initialize the optional attributes with
        default values.
        """
        required = {
            'color': 0.25,
        }
        obj = p.Solid(**required)
        for attr in required:
            assert getattr(obj, attr) == required[attr]

    # Tests for fill.
    def test_fill(self):
        """Given origin, dimensions, and a color, :meth:`Solid.fill`
        should return a volume filled with a box of the origin,
        dimensions, and color given when the object was created.
        """
        obj = p.Solid(color=0x40 / 0xff)
        result = obj.fill((2, 4, 4))
        assert (mkhex(result) == np.array([
            [
                [0x40, 0x40, 0x40, 0x40],
                [0x40, 0x40, 0x40, 0x40],
                [0x40, 0x40, 0x40, 0x40],
                [0x40, 0x40, 0x40, 0x40],
            ],
            [
                [0x40, 0x40, 0x40, 0x40],
                [0x40, 0x40, 0x40, 0x40],
                [0x40, 0x40, 0x40, 0x40],
                [0x40, 0x40, 0x40, 0x40],
            ],
        ], dtype=np.uint8)).all()


class TestSpheres:
    # Tests for initialization.
    def test_init_all_default(self):
        """Given only required parameters, :class:`Spheres` should
        initialize the required attributes with the given values.
        It should then initialize the optional attributes with
        default values.
        """
        required = {
            'radius': 2.0,
        }
        optional = {
            'offset': '',
            'cells': False,
            'round': True,
        }
        obj = p.Spheres(**required)
        for attr in required:
            assert getattr(obj, attr) == required[attr]
        for attr in optional:
            assert getattr(obj, attr) == optional[attr]

    def test_init_all_optional(self):
        """Given optional parameters, :class:`Spheres` should
        initialize the given attributes with the given values.
        """
        required = {
            'radius': 2.0,
        }
        optional = {
            'offset': 'x',
            'cells': True,
            'round': False,
        }
        obj = p.Spheres(**required, **optional)
        for attr in required:
            assert getattr(obj, attr) == required[attr]
        for attr in optional:
            assert getattr(obj, attr) == optional[attr]

    # Tests for fill.
    def test_fill_x(self):
        """Given the shape of the output, :meth:`Spheres.fill`
        should return a volume filled with a box of the origin,
        dimensions, and color given when the object was created.
        """
        obj = p.Spheres(radius=5, offset='x')
        result = obj.fill((1, 8, 8))
        assert (mkhex(result) == np.array([
            [
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x72, 0x90, 0x98, 0x90, 0x72],
                [0x00, 0x00, 0x86, 0xb0, 0xc5, 0xcc, 0xc5, 0xb0],
                [0x00, 0x72, 0xb0, 0xd2, 0xe4, 0xe9, 0xe4, 0xd2],
                [0x00, 0x90, 0xc5, 0xe4, 0xf4, 0xf9, 0xf4, 0xe4],
                [0x00, 0x98, 0xcc, 0xe9, 0xf9, 0xff, 0xf9, 0xe9],
                [0x00, 0x90, 0xc5, 0xe4, 0xf4, 0xf9, 0xf4, 0xe4],
                [0x00, 0x72, 0xb0, 0xd2, 0xe4, 0xe9, 0xe4, 0xd2],
            ],
        ], dtype=np.uint8)).all()

    def test_fill_y(self):
        """Given the shape of the output, :meth:`Spheres.fill`
        should return a volume filled with a box of the origin,
        dimensions, and color given when the object was created.
        """
        obj = p.Spheres(radius=5, offset='y')
        result = obj.fill((1, 8, 8))
        assert (mkhex(result) == np.array([
            [
                [0x00, 0x98, 0xcc, 0xe9, 0xf9, 0xff, 0xf9, 0xe9],
                [0x00, 0x90, 0xc5, 0xe4, 0xf4, 0xf9, 0xf4, 0xe4],
                [0x00, 0x72, 0xb0, 0xd2, 0xe4, 0xe9, 0xe4, 0xd2],
                [0x00, 0x00, 0x86, 0xb0, 0xc5, 0xcc, 0xc5, 0xb0],
                [0x00, 0x00, 0x00, 0x72, 0x90, 0x98, 0x90, 0x72],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x72, 0x90, 0x98, 0x90, 0x72],
                [0x00, 0x00, 0x86, 0xb0, 0xc5, 0xcc, 0xc5, 0xb0],
            ],
        ], dtype=np.uint8)).all()


class TestSpot:
    # Tests for initialization.
    def test_init_all_default(self):
        """Given only required parameters, :class:`Spot` should
        initialize the required attributes with the given values.
        It should then initialize the optional attributes with
        default values.
        """
        required = {
            'radius': 2.0,
        }
        optional = {}
        obj = p.Spot(**required)
        for attr in required:
            assert getattr(obj, attr) == required[attr]
        for attr in optional:
            assert getattr(obj, attr) == optional[attr]

    def test_init_all_optional(self):
        """Given optional parameters, :class:`Spot` should
        initialize the given attributes with the given values.
        """
        required = {
            'radius': 2.0,
        }
        optional = {}
        obj = p.Spot(**required, **optional)
        for attr in required:
            assert getattr(obj, attr) == required[attr]
        for attr in optional:
            assert getattr(obj, attr) == optional[attr]

    # Tests for fill.
    def test_fill(self):
        """Given the shape of an output array, :meth:`Spot.fill`
        should return a volume filled with a box of the origin,
        dimensions, and color given when the object was created.
        """
        obj = p.Spot(radius=5)
        result = obj.fill((1, 8, 8))
        assert (mkhex(result) == np.array([
            [
                [0x4c, 0x63, 0x75, 0x7f, 0x7f, 0x75, 0x63, 0x4c],
                [0x63, 0x7f, 0x95, 0xa3, 0xa3, 0x95, 0x7f, 0x63],
                [0x75, 0x95, 0xb2, 0xc5, 0xc5, 0xb2, 0x95, 0x75],
                [0x7f, 0xa3, 0xc5, 0xe5, 0xe5, 0xc5, 0xa3, 0x7f],
                [0x7f, 0xa3, 0xc5, 0xe5, 0xe5, 0xc5, 0xa3, 0x7f],
                [0x75, 0x95, 0xb2, 0xc5, 0xc5, 0xb2, 0x95, 0x75],
                [0x63, 0x7f, 0x95, 0xa3, 0xa3, 0x95, 0x7f, 0x63],
                [0x4c, 0x63, 0x75, 0x7f, 0x7f, 0x75, 0x63, 0x4c],
            ],
        ], dtype=np.uint8)).all()


class TestText:
    def test_fill(self):
        """Given the shape of an output array, :meth:`Spot.fill`
        should return a volume filled with a box of the origin,
        dimensions, and color given when the object was created.
        """
        obj = p.Text(text='s', size=6, origin=(3, 0))
        result = obj.fill((1, 8, 8))
        assert (mkhex(result) == np.array([
            [
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x0b, 0x50, 0x2c, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x8e, 0x33, 0x3c, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x29, 0x8a, 0x74, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x61, 0x6f, 0x8a, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00],
            ],
        ], dtype=np.uint8)).all()


class TestWaves:
    def test_fill(self):
        """Given the shape of an output array, :meth:`Waves.fill`
        should return a volume filled a cosine wave.
        """
        obj = p.Waves(unit=7)
        result = obj.fill((1, 8, 8))
        assert (mkhex(result) == np.array([
            [
                [0xff, 0xce, 0x63, 0x0c, 0x0c, 0x63, 0xce, 0xff],
                [0xff, 0xce, 0x63, 0x0c, 0x0c, 0x63, 0xce, 0xff],
                [0xff, 0xce, 0x63, 0x0c, 0x0c, 0x63, 0xce, 0xff],
                [0xff, 0xce, 0x63, 0x0c, 0x0c, 0x63, 0xce, 0xff],
                [0xff, 0xce, 0x63, 0x0c, 0x0c, 0x63, 0xce, 0xff],
                [0xff, 0xce, 0x63, 0x0c, 0x0c, 0x63, 0xce, 0xff],
                [0xff, 0xce, 0x63, 0x0c, 0x0c, 0x63, 0xce, 0xff],
                [0xff, 0xce, 0x63, 0x0c, 0x0c, 0x63, 0xce, 0xff],
            ],
        ], dtype=np.uint8)).all()

    def test_fill_with_radial(self):
        """Given the shape of an output array, :meth:`Waves.fill`
        should return a volume filled a cosine wave. If radial was set,
        the waves should emanate from the center rather then from along
        the side.
        """
        obj = p.Waves(unit=7, radial=True)
        result = obj.fill((1, 8, 8))
        assert (mkhex(result) == np.array([
            [
                [0x5d, 0x1f, 0x04, 0x00, 0x00, 0x04, 0x1f, 0x5d],
                [0x1f, 0x00, 0x11, 0x2b, 0x2b, 0x11, 0x00, 0x1f],
                [0x04, 0x11, 0x55, 0x92, 0x92, 0x55, 0x11, 0x04],
                [0x00, 0x2b, 0x92, 0xe6, 0xe6, 0x92, 0x2b, 0x00],
                [0x00, 0x2b, 0x92, 0xe6, 0xe6, 0x92, 0x2b, 0x00],
                [0x04, 0x11, 0x55, 0x92, 0x92, 0x55, 0x11, 0x04],
                [0x1f, 0x00, 0x11, 0x2b, 0x2b, 0x11, 0x00, 0x1f],
                [0x5d, 0x1f, 0x04, 0x00, 0x00, 0x04, 0x1f, 0x5d],
            ],
        ], dtype=np.uint8)).all()

    def test_fill_with_slope(self):
        """Given the shape of an output array, :meth:`Waves.fill`
        should return a volume filled a cosine wave. If angle was set,
        the waves should be slanted by that amount.
        """
        obj = p.Waves(unit=7, angle=45)
        result = obj.fill((1, 8, 8))
        assert (mkhex(result) == np.array([
            [
                [0xff, 0xf2, 0xce, 0x9b, 0x63, 0x30, 0x0c, 0x00],
                [0xf2, 0xce, 0x9b, 0x63, 0x30, 0x0c, 0x00, 0x0c],
                [0xce, 0x9b, 0x63, 0x30, 0x0c, 0x00, 0x0c, 0x30],
                [0x9b, 0x63, 0x30, 0x0c, 0x00, 0x0c, 0x30, 0x63],
                [0x63, 0x30, 0x0c, 0x00, 0x0c, 0x30, 0x63, 0x9b],
                [0x30, 0x0c, 0x00, 0x0c, 0x30, 0x63, 0x9b, 0xce],
                [0x0c, 0x00, 0x0c, 0x30, 0x63, 0x9b, 0xce, 0xf2],
                [0x00, 0x0c, 0x30, 0x63, 0x9b, 0xce, 0xf2, 0xff],
            ],
        ], dtype=np.uint8)).all()

    def test_fill_with_warp(self):
        """Given the shape of an output array, :meth:`Waves.fill`
        should return a volume filled a cosine wave. If warp was set,
        the underlying space should be warped by the warp function.
        """
        def warp(a):
            return a + 0.25
        
        obj = p.Waves(unit=7, warp=warp)
        result = obj.fill((1, 8, 8))
        assert (mkhex(result) == np.array([
            [
                [0x7f, 0x1b, 0x03, 0x48, 0xb6, 0xfb, 0xe3, 0x7f],
                [0x7f, 0x1b, 0x03, 0x48, 0xb6, 0xfb, 0xe3, 0x7f],
                [0x7f, 0x1b, 0x03, 0x48, 0xb6, 0xfb, 0xe3, 0x7f],
                [0x7f, 0x1b, 0x03, 0x48, 0xb6, 0xfb, 0xe3, 0x7f],
                [0x7f, 0x1b, 0x03, 0x48, 0xb6, 0xfb, 0xe3, 0x7f],
                [0x7f, 0x1b, 0x03, 0x48, 0xb6, 0xfb, 0xe3, 0x7f],
                [0x7f, 0x1b, 0x03, 0x48, 0xb6, 0xfb, 0xe3, 0x7f],
                [0x7f, 0x1b, 0x03, 0x48, 0xb6, 0xfb, 0xe3, 0x7f],
            ],
        ], dtype=np.uint8)).all()

    def test_fill_with_wavelength(self):
        """Given the shape of an output array, :meth:`Waves.fill`
        should return a volume filled a cosine wave. If wavelength was
        set, the wave should have peaks that match the wavelength.
        """
        obj = p.Waves(unit=7, wavelength=0.5)
        result = obj.fill((1, 8, 8))
        assert (mkhex(result) == np.array([
            [
                [0xff, 0x63, 0x0c, 0xce, 0xce, 0x0c, 0x63, 0xff],
                [0xff, 0x63, 0x0c, 0xce, 0xce, 0x0c, 0x63, 0xff],
                [0xff, 0x63, 0x0c, 0xce, 0xce, 0x0c, 0x63, 0xff],
                [0xff, 0x63, 0x0c, 0xce, 0xce, 0x0c, 0x63, 0xff],
                [0xff, 0x63, 0x0c, 0xce, 0xce, 0x0c, 0x63, 0xff],
                [0xff, 0x63, 0x0c, 0xce, 0xce, 0x0c, 0x63, 0xff],
                [0xff, 0x63, 0x0c, 0xce, 0xce, 0x0c, 0x63, 0xff],
                [0xff, 0x63, 0x0c, 0xce, 0xce, 0x0c, 0x63, 0xff],
            ],
        ], dtype=np.uint8)).all()
