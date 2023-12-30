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
            'drop': 0.0,
            'seed': None,
            'seed_img': None,
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
            'drop': 0.25,
            'seed': 'spam',
            'seed_img': np.ones((1, 8, 8), dtype=float),
        }
        obj = t.Tile(**required, **optional)
        for attr in required:
            assert getattr(obj, attr) == required[attr]
        for attr in optional:
            if not isinstance(optional[attr], np.ndarray):
                assert getattr(obj, attr) == optional[attr]
            else:
                assert (getattr(obj, attr) == optional[attr]).all()
    
    def test_fill_triangle(self):
        """Given a size for image data, :meth:`Tile.fill` should
        return a volume of image data filled with the tile pattern
        given by the attributes of the :class:`Tile`.
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
