"""
test_convert
~~~~~~~~~~~~

Unit tests for :mod:`pjimg.util.convert`.
"""
import numpy as np
import pytest as pt

import pjimg.util.convert as cvrt


# Test cases.
class TestFloatToUint8:
    def test_float_to_uint8_convert(self):
        """Given an array-like object of floating point values
        between zero and one, :func:`float_to_uint8` should return
        a :class:`numpy.ndarray` object of unsigned 8-bit integers
        between zero and 255.
        """
        a = np.array([[
            [0., .5, 1.,],
            [0., .5, 1.,],
            [0., .5, 1.,],
        ],])
        assert (cvrt.float_to_uint8(a) == np.array([
            [
                [0x00, 0x7f, 0xff,],
                [0x00, 0x7f, 0xff,],
                [0x00, 0x7f, 0xff,],
            ],
        ], dtype=np.uint8)).all()

    def test_float_to_uint8_invalid(self):
        """Given an array-like object of floating point with a value
        greater than one, :func:`float_to_uint8` should raise a
        :class:`ValueError` exception.
        """
        a = np.array([[
            [0., .5, 1.1,],
            [0., .5, 1.,],
            [0., .5, 1.,],
        ],])
        with pt.raises(ValueError, match='Array values must be 0 >= x >= 1.'):
            _ = cvrt.float_to_uint8(a)
