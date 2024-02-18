"""
test_resize
~~~~~~~~~~~

Unit tests for the lerpy.resize module.
"""
import numpy as np
import pytest as pt

from pjimg.util import lerps as lp
from pjimg.util import resize as rs
from tests.fixtures import a


# Test classes.
class TestCropArray:
    def test_crop_centered(self, a):
        """Given an array and a final size, :func:`crop_array` should
        crop out a section of the array of the final size from the
        center of the array.
        """
        size = (3, 3)
        assert (rs.crop_array(a, size) == np.array([
            [0.50, 0.75, 1.00,],
            [0.75, 1.00, 0.75,],
            [1.00, 0.75, 0.50,],
        ])).all()

    def test_crop_offset(self, a):
        """Given an array, a final size, and an offset location,
        :func:`crop_array` should crop out a section of the array of
        the final size from a location offset center of the array by
        the offset location.
        """
        size = (3, 3)
        loc = (-1, -1)
        assert (rs.crop_array(a, size, loc) == np.array([
            [0.00, 0.25, 0.50],
            [0.25, 0.50, 0.75],
            [0.50, 0.75, 1.00],
        ])).all()


# Tests for build_resizing_matrices.
def test_build_resizing_matrix_increase_size():
    """Given an original size and a final size,
    :funct:`build_resizing_matrices` should
    return two arrays of indexes and one of
    distances that can be used to interpolate
    the values of the resized array.
    """
    src_shape = (3, 3)
    dst_shape = (5, 5)
    act_a, act_b, act_x = rs.build_resizing_matrices(src_shape, dst_shape)
    assert (act_a == np.array([
        [
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2],
            ],
            [
                [0, 0, 1, 1, 2],
                [0, 0, 1, 1, 2],
                [0, 0, 1, 1, 2],
                [0, 0, 1, 1, 2],
                [0, 0, 1, 1, 2],
            ],
        ],
        [
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2],
                [2, 2, 2, 2, 2],
                [2, 2, 2, 2, 2],
            ],
            [
                [0, 0, 1, 1, 2],
                [0, 0, 1, 1, 2],
                [0, 0, 1, 1, 2],
                [0, 0, 1, 1, 2],
                [0, 0, 1, 1, 2],
            ],
        ],
    ])).all()
    assert (act_b == np.array([
        [
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2],
            ],
            [
                [1, 1, 2, 2, 2],
                [1, 1, 2, 2, 2],
                [1, 1, 2, 2, 2],
                [1, 1, 2, 2, 2],
                [1, 1, 2, 2, 2],
            ],
        ],
        [
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2],
                [2, 2, 2, 2, 2],
                [2, 2, 2, 2, 2],
            ],
            [
                [1, 1, 2, 2, 2],
                [1, 1, 2, 2, 2],
                [1, 1, 2, 2, 2],
                [1, 1, 2, 2, 2],
                [1, 1, 2, 2, 2],
            ],
        ],
    ])).all()
    assert (act_x == np.array([
        [
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        ],
        [
            [0.0000, 0.5000, 0.0000, 0.5000, 0.0000],
            [0.0000, 0.5000, 0.0000, 0.5000, 0.0000],
            [0.0000, 0.5000, 0.0000, 0.5000, 0.0000],
            [0.0000, 0.5000, 0.0000, 0.5000, 0.0000],
            [0.0000, 0.5000, 0.0000, 0.5000, 0.0000],
        ],
    ])).all()


# Tests for magnify_size.
def test_magnify_size():
    """Given the shape of an array and a magnification factor,
    :funct:`magnify_size` should return the magnified shape of
    the array.
    """
    assert rs.magnify_size((5, 5, 5), 2.0) == (10, 10, 10)


# Tests for resize_array.
def test_resize_array_three_dimensions():
    """Given a three-dimensional array and a new size,
    :funct:`resize_array` should return an array of
    the new size with the data resized through trilinear
    interpolation.
    """
    a = np.array([
        [
            [0.0, 1.0, 2.0, ],
            [1.0, 2.0, 3.0, ],
            [2.0, 3.0, 4.0, ],
        ],
        [
            [1.0, 2.0, 3.0, ],
            [2.0, 3.0, 4.0, ],
            [3.0, 4.0, 5.0, ],
        ],
        [
            [2.0, 3.0, 4.0, ],
            [3.0, 4.0, 5.0, ],
            [4.0, 5.0, 6.0, ],
        ],
    ])
    assert (rs.resize_array(a, (5, 5, 5)) == np.array([
        [
            [0.0, 0.5, 1.0, 1.5, 2.0],
            [0.5, 1.0, 1.5, 2.0, 2.5],
            [1.0, 1.5, 2.0, 2.5, 3.0],
            [1.5, 2.0, 2.5, 3.0, 3.5],
            [2.0, 2.5, 3.0, 3.5, 4.0],
        ],
        [
            [0.5, 1.0, 1.5, 2.0, 2.5],
            [1.0, 1.5, 2.0, 2.5, 3.0],
            [1.5, 2.0, 2.5, 3.0, 3.5],
            [2.0, 2.5, 3.0, 3.5, 4.0],
            [2.5, 3.0, 3.5, 4.0, 4.5],
        ],
        [
            [1.0, 1.5, 2.0, 2.5, 3.0],
            [1.5, 2.0, 2.5, 3.0, 3.5],
            [2.0, 2.5, 3.0, 3.5, 4.0],
            [2.5, 3.0, 3.5, 4.0, 4.5],
            [3.0, 3.5, 4.0, 4.5, 5.0],
        ],
        [
            [1.5, 2.0, 2.5, 3.0, 3.5],
            [2.0, 2.5, 3.0, 3.5, 4.0],
            [2.5, 3.0, 3.5, 4.0, 4.5],
            [3.0, 3.5, 4.0, 4.5, 5.0],
            [3.5, 4.0, 4.5, 5.0, 5.5],
        ],
        [
            [2.0, 2.5, 3.0, 3.5, 4.0],
            [2.5, 3.0, 3.5, 4.0, 4.5],
            [3.0, 3.5, 4.0, 4.5, 5.0],
            [3.5, 4.0, 4.5, 5.0, 5.5],
            [4.0, 4.5, 5.0, 5.5, 6.0],
        ],
    ])).all()


def test_resize_array_two_dimensions():
    """Given a two-dimensional array and a new size,
    :funct:`resize_array` should return an array of
    the new size with the data resized through
    bilinear interpolation.
    """
    a = np.array([
        [0.0, 1.0, 2.0, ],
        [1.0, 2.0, 3.0, ],
        [2.0, 3.0, 4.0, ],
    ])
    size = (5, 5)
    assert (rs.resize_array(a, size) == np.array([
        [0.0, 0.5, 1.0, 1.5, 2.0, ],
        [0.5, 1.0, 1.5, 2.0, 2.5, ],
        [1.0, 1.5, 2.0, 2.5, 3.0, ],
        [1.5, 2.0, 2.5, 3.0, 3.5, ],
        [2.0, 2.5, 3.0, 3.5, 4.0, ],
    ])).all()


def test_resize_array_two_dimensions_not_square():
    """Given a two-dimensional array and a new size,
    :funct:`resize_array` should return an array of
    the new size with the data resized through
    bilinear interpolation. If the size of the X dimension
    of the array does not equal the Y dimension of the
    array, the resizing should still work as expected.
    """
    a = np.array([
        [0.0, 1.0, 2.0, ],
        [1.0, 2.0, 3.0, ],
        [2.0, 3.0, 4.0, ],
        [3.0, 4.0, 5.0, ]
    ])
    size = (7, 5)
    exp = rs.resize_array(a, size)
    assert exp.shape == size
    assert (exp == np.array([
        [0.0, 0.5, 1.0, 1.5, 2.0, ],
        [0.5, 1.0, 1.5, 2.0, 2.5, ],
        [1.0, 1.5, 2.0, 2.5, 3.0, ],
        [1.5, 2.0, 2.5, 3.0, 3.5, ],
        [2.0, 2.5, 3.0, 3.5, 4.0, ],
        [2.5, 3.0, 3.5, 4.0, 4.5, ],
        [3.0, 3.5, 4.0, 4.5, 5.0, ],
    ])).all()


def test_resize_array_shrink_array():
    """If the new size is smaller than the original size,
    :funct:`resize_array` should return a smaller array.
    """
    a = np.array([
        [1, 2, 3, 4, 5],
        [2, 3, 3, 3, 4],
        [3, 3, 3, 3, 3],
        [4, 3, 3, 3, 2],
        [5, 4, 3, 2, 1],
    ])
    size = (3, 3)
    assert (rs.resize_array(a, size) == np.array([
        [1, 3, 5],
        [3, 3, 3],
        [5, 3, 1],
    ])).all()


def test_resize_array_ndcerp():
    """Given the n-dimensional cubic interpolation function,
    :funct:`resize_array` should use it rather than the
    n-dimensional linear interpolation function.
    """
    a = np.arange(9, dtype=float).reshape((3, 3))
    a = a ** 2
    size = (5, 5)
    erp = lp.ndcerp

    act = rs.resize_array(a, size, erp)
    act = np.around(act, 4)
    assert (act == np.array([
        [0.0000, 0.3125, 1.0000, 2.5000, 4.0000],
        [4.3164, 5.8906, 8.2617, 11.3125, 14.5938],
        [9.0000, 11.9375, 16.0000, 20.5000, 25.0000],
        [22.1523, 26.4688, 32.2852, 38.3125, 44.7812],
        [36.0000, 41.5625, 49.0000, 56.5000, 64.0000],
    ])).all()
