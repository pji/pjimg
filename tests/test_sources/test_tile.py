"""
test_tile
~~~~~~~~~

Unit tests for :mod:`pjimg.sources.tile`.
"""
import numpy as np

from pjimg.sources import tile as t
from tests.common import mkhex


class TestTile:
    def test_init_all_default(self):
        """Given only required parameters, :class:`Tile` should
        initialize the required attributes with the given values. It
        should then initialize the optional attributes with default
        values.
        """
        required = {
            'pattern': 'triangle',
            'radius': 3,
            'gap': 1,
        }
        optional = {
            'rotation': 0.0,
            'color': 1.0,
            'color_img': None,
            'drop': 0.0,
            'seed': None,
            'drop_img': None,
        }
        obj = t.Tile(**required)
        for attr in required:
            assert getattr(obj, attr) == required[attr]
        for attr in optional:
            assert getattr(obj, attr) == optional[attr]
    
    def test_init_all_optional(self):
        """Given optional parameters, :class:`Tile` should
        initialize those attributes with the given values.
        """
        required = {
            'pattern': 'triangle',
            'radius': 3,
            'gap': 1,
        }
        optional = {
            'rotation': np.pi / 2,
            'color': 0.5,
            'color_img': np.zeros((1, 8, 8), dtype=float),
            'drop': 0.25,
            'seed': 'spam',
            'drop_img': np.ones((1, 8, 8), dtype=float),
        }
        obj = t.Tile(**required, **optional)
        for attr in required:
            assert getattr(obj, attr) == required[attr]
        for attr in optional:
            if not isinstance(optional[attr], np.ndarray):
                assert getattr(obj, attr) == optional[attr]
            else:
                assert (getattr(obj, attr) == optional[attr]).all()

    def test_fill_color_img(self):
        """Given a size for image data, :meth:`Tile.fill` should
        return a volume of image data filled with the tile pattern
        given by the attributes of the :class:`Tile`. If `color_img`
        is set on the object, the color of the tiles in the final
        image should be based on the colors in the `color_img`.
        """
        pattern = 'triangle'
        radius = 2
        gap = 1
        color_img = np.array([[
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
        ],], dtype=float)
        obj = t.Tile(pattern, radius, gap, color_img=color_img)
        result = obj.fill((1, 8, 8))
        assert (mkhex(result) == np.array([[
            [0x59, 0xbf, 0xbf, 0x99, 0x13, 0x40, 0x53, 0x53],
            [0xa4, 0xc0, 0xc0, 0xb1, 0x4d, 0x53, 0x53, 0x53],
            [0xc4, 0xc4, 0xb8, 0x89, 0x51, 0x51, 0x51, 0x51],
            [0xa8, 0xc4, 0xa8, 0x91, 0x62, 0x51, 0x51, 0x51],
            [0x6b, 0xc0, 0x97, 0x91, 0x8d, 0x52, 0x51, 0x3d],
            [0x32, 0xbe, 0x82, 0x91, 0x91, 0x83, 0x48, 0x2e],
            [0xe5, 0x13, 0x91, 0x91, 0x91, 0x91, 0x19, 0x33],
            [0x2e, 0xc8, 0x84, 0x91, 0x91, 0x5b, 0x4c, 0x24],
        ]], dtype=np.uint8)).all()
    
    def test_fill_drop(self):
        """Given a size for image data, :meth:`Tile.fill` should
        return a volume of image data filled with the tile pattern
        given by the attributes of the :class:`Tile`. If drop is
        set on the object, that percentage of tiles should be
        dropped from the pattern.
        """
        pattern = 'triangle'
        radius = 2
        gap = 1
        drop = 0.5
        seed = 'spam'
        obj = t.Tile(pattern, radius, gap, drop=drop, seed=seed)
        result = obj.fill((1, 8, 8))
        assert (mkhex(result) == np.array([[
            [0x77, 0xff, 0xff, 0x3f, 0x00, 0x00, 0x00, 0x00],
            [0xd6, 0xff, 0xff, 0xff, 0x3d, 0x33, 0x35, 0x33],
            [0x1f, 0x3a, 0x64, 0xff, 0xff, 0xff, 0xff, 0xff],
            [0x00, 0x00, 0x8b, 0xff, 0xf7, 0xff, 0xff, 0xff],
            [0x00, 0x10, 0xdb, 0xff, 0xff, 0xf7, 0xff, 0xf5],
            [0x00, 0x1d, 0xd7, 0xff, 0xff, 0xff, 0xeb, 0xe4],
            [0x00, 0x1b, 0xff, 0xff, 0xff, 0xff, 0x2f, 0x23],
            [0x36, 0xff, 0xe4, 0xff, 0xff, 0xbb, 0xff, 0x40],
        ]], dtype=np.uint8)).all()
    
    def test_fill_drop_img(self):
        """Given a size for image data, :meth:`Tile.fill` should
        return a volume of image data filled with the tile pattern
        given by the attributes of the :class:`Tile`. If `drop_img`
        is set on the object, a percentage of tiles should be
        dropped from the pattern based on the seed image.
        """
        pattern = 'triangle'
        radius = 2
        gap = 1
        drop_img = np.array([[
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
        ],], dtype=float)
        seed = 'spam'
        obj = t.Tile(pattern, radius, gap, drop_img=drop_img, seed=seed)
        result = obj.fill((1, 8, 8))
        assert (mkhex(result) == np.array([[
            [0x77, 0xff, 0xff, 0x3f, 0x00, 0x00, 0x00, 0x00],
            [0xd6, 0xff, 0xff, 0xff, 0x28, 0x00, 0x00, 0x00],
            [0x1f, 0x3a, 0x64, 0xff, 0x40, 0x00, 0x00, 0x00],
            [0x00, 0x00, 0x8b, 0xff, 0xe7, 0x12, 0x00, 0x00],
            [0x00, 0x10, 0xdb, 0xff, 0xff, 0xa7, 0x00, 0x00],
            [0x04, 0x1d, 0xd7, 0xff, 0xff, 0xff, 0x23, 0x00],
            [0xff, 0x1b, 0xff, 0xff, 0xff, 0xff, 0x28, 0x00],
            [0x3a, 0xff, 0xe4, 0xff, 0xff, 0xbb, 0xff, 0x40],
        ]], dtype=np.uint8)).all()
    
    def test_fill_hexagon(self):
        """Given a size for image data, :meth:`Tile.fill` should
        return a volume of image data filled with the tile pattern
        given by the attributes of the :class:`Tile`. If the pattern
        is `hexagon`, the tiles should be hexagon.
        """
        pattern = 'hexagon'
        radius = 2
        gap = 1
        obj = t.Tile(pattern, radius, gap)
        result = obj.fill((1, 8, 8))
        assert (mkhex(result) == np.array([[
            [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff],
            [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff],
            [0xff, 0xff, 0xd6, 0xc2, 0xff, 0xff, 0xff, 0xc5],
            [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff],
            [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff],
            [0xff, 0xff, 0xff, 0xff, 0xff, 0xf2, 0xff, 0xff],
            [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff],
            [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff],
        ]], dtype=np.uint8)).all()

    def test_fill_octagon(self):
        """Given a size for image data, :meth:`Tile.fill` should
        return a volume of image data filled with the tile pattern
        given by the attributes of the :class:`Tile`. If the pattern
        is `octagon`, the tiles should be octagon.
        """
        pattern = 'octagon'
        radius = 2
        gap = 1
        obj = t.Tile(pattern, radius, gap)
        result = obj.fill((1, 8, 8))
        assert (mkhex(result) == np.array([[
            [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff],
            [0xff, 0xff, 0xff, 0x69, 0x66, 0xff, 0xff, 0xff],
            [0xff, 0xff, 0x4c, 0xff, 0xff, 0x4f, 0xff, 0xff],
            [0xff, 0x67, 0xff, 0xff, 0xff, 0xff, 0x55, 0x3e],
            [0xff, 0x68, 0xff, 0xff, 0xff, 0xff, 0x57, 0x33],
            [0xff, 0xff, 0x4f, 0xff, 0xff, 0x4c, 0xff, 0xff],
            [0xff, 0xff, 0xff, 0x56, 0x56, 0xff, 0xff, 0xff],
            [0xff, 0xff, 0xff, 0x3d, 0x36, 0xff, 0xff, 0xff],
        ]], dtype=np.uint8)).all()

    def test_fill_octagonwithsquares(self):
        """Given a size for image data, :meth:`Tile.fill` should
        return a volume of image data filled with the tile pattern
        given by the attributes of the :class:`Tile`. If the pattern
        is `octagonwithsquares`, the tiles should be octagons and
        squares.
        """
        pattern = 'octagonwithsquares'
        radius = 2
        gap = 1
        obj = t.Tile(pattern, radius, gap)
        result = obj.fill((1, 8, 8))
        assert (mkhex(result) == np.array([[
            [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff],
            [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff],
            [0xff, 0xff, 0x4c, 0xff, 0xff, 0x4f, 0xff, 0xff],
            [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff],
            [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff],
            [0xff, 0xff, 0x4f, 0xff, 0xff, 0x4c, 0xff, 0xff],
            [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff],
            [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff],
        ]], dtype=np.uint8)).all()

    def test_fill_square(self):
        """Given a size for image data, :meth:`Tile.fill` should
        return a volume of image data filled with the tile pattern
        given by the attributes of the :class:`Tile`. If the pattern
        is `square`, the tiles should be squares.
        """
        pattern = 'square'
        radius = 2
        gap = 1
        obj = t.Tile(pattern, radius, gap)
        result = obj.fill((1, 8, 8))
        assert (mkhex(result) == np.array([[
            [0x25, 0xff, 0xca, 0x23, 0xff, 0xb8, 0xff, 0xba],
            [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff],
            [0xc0, 0xff, 0xff, 0x7b, 0xff, 0xff, 0xff, 0xff],
            [0xff, 0xff, 0x78, 0xff, 0xff, 0xff, 0xff, 0x7b],
            [0x43, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff],
            [0xb8, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff],
            [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff],
            [0xba, 0xff, 0xff, 0x81, 0xff, 0xff, 0xff, 0xff],
        ]], dtype=np.uint8)).all()

    def test_fill_triangle(self):
        """Given a size for image data, :meth:`Tile.fill` should
        return a volume of image data filled with the tile pattern
        given by the attributes of the :class:`Tile`. If the pattern
        is `triangle`, the tiles should be triangles.
        """
        pattern = 'triangle'
        radius = 2
        gap = 1
        obj = t.Tile(pattern, radius, gap)
        result = obj.fill((1, 8, 8))
        assert (mkhex(result) == np.array([[
            [0x77, 0xff, 0xff, 0xff, 0x3b, 0xc7, 0xff, 0xff],
            [0xda, 0xff, 0xff, 0xff, 0xe9, 0xff, 0xff, 0xff],
            [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff],
            [0xda, 0xff, 0xff, 0xff, 0xf7, 0xff, 0xff, 0xff],
            [0x8c, 0xff, 0xfb, 0xff, 0xff, 0xf7, 0xff, 0xf5],
            [0x41, 0xff, 0xe2, 0xff, 0xff, 0xff, 0xeb, 0xe6],
            [0xff, 0x1f, 0xff, 0xff, 0xff, 0xff, 0x48, 0xff],
            [0x3a, 0xff, 0xe4, 0xff, 0xff, 0xbb, 0xff, 0xa7],
        ]], dtype=np.uint8)).all()
