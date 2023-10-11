"""
test_lerp
~~~~~~~~~

Unit tests for :mod:`pjimg.util.lerp`.
"""
from math import prod

import numpy as np
import pytest as pt

from pjimg.util import lerps as lp


# Tests for cubic_interpolation.
def test_cubic_interpolation():
    """Given four arrays of values named a, b, c, and d and an
    array of distances between the points in the b and c arrays,
    :funct:`cubic_interpolation` should return an array of the
    interpolated values.
    """
    a_ = np.array([-1.0, 0.0, 1.0, 4.0])
    a = np.array([0.0, 1.0, 4.0, 9.0])
    b = np.array([1.0, 4.0, 9.0, 16.0])
    b_ = np.array([4.0, 9.0, 16.0, 25.0])
    x = np.array([0.5, 0.5, 0.5, 0.5])
    assert (lp.cubic_interpolation(a, b, x, a_, b_) == np.array(
        [0.3750, 2.2500, 6.2500, 12.2500]
    )).all()


def test_cubic_interpolation_without_primes():
    """Given two arrays of values named a, b, and an
    array of distances between the points in two arrays,
    :funct:`cubic_interpolation` should return an array
    of the interpolated values.
    """
    a = np.array([0.0, 1.0, 4.0, 9.0])
    b = np.array([1.0, 4.0, 9.0, 16.0])
    x = np.array([0.5, 0.5, 0.5, 0.5])
    assert (lp.cubic_interpolation(a, b, x) == np.array(
        [0.3125, 2.2500, 6.2500, 12.8125])
    ).all()


# Tests for linear_interpolation.
def test_linear_interpolation():
    """Given two linear arrays of values and one linear array of
    distances, :funct:`linear_interpolation` should return an array
    that is the linear interpolation of the values at the distances.
    """
    a = np.array([0.0, 1.0, 2.0, 3.0, 4.0,])
    b = np.array([1.0, 2.0, 3.0, 4.0, 5.0,])
    x = np.array([0.5, 0.5, 0.5, 0.5, 0.5,])
    assert (lp.linear_interpolation(a, b, x) == np.array(
        [0.5, 1.5, 2.5, 3.5, 4.5,]
    )).all()


def test_perserving_array_data_type():
    """The returned array from :funct:`linear_interpolation` should
    have the same datatype as the first of the given value arrays.
    """
    a = np.array([0, 1, 2, 3], dtype=np.uint8)
    b = np.array([1, 2, 3, 4], dtype=np.uint8)
    x = np.array([0.5, 0.5, 0.5, 0.5], dtype=float)
    result = lp.linear_interpolation(a, b, x)
    assert result.dtype == np.uint8


# Tests for n_dimensional_interpolation
def test_nderp_wrong_number_of_points():
    """If the passed values do not contain enough points
    to perform the interpolation over multiple dimensions,
    :funct:`n_dimensional_interpolation` should raise
    a :class:`ValueError` exception.
    """
    size = (3, 3, 3)
    dims = len(size)
    ab_shape = tuple([2 * dims // 2, *size])      # Should be 2 ** dims.
    length = prod(ab_shape)
    a = np.arange(length).reshape(ab_shape)
    b = a ** 2
    x = np.full((dims, *size), .5)

    with pt.raises(
        ValueError,
        match='Not the correct number of points for the dimensions.'
    ):
        lp.n_dimensional_interpolation(a, b, x, lp.lerp)


# Tests for n_dimensional_cubic_interpolation.
def test_ndcerp_two_dimensions():
    '''Given two arrays of values and an array of distances,
    :funct:`n_dimensional_cubic_interpolation` should return
    an array with the cubic interpolation of the value arrays.
    '''
    size = (3, 3)
    dims = len(size)
    ab_shape = tuple([2 ** dims // 2, *size])
    length = prod(ab_shape)
    a = np.arange(length).reshape(ab_shape)
    b = a ** 2
    x = np.full((dims, *size), .5)
    assert (lp.n_dimensional_cubic_interpolation(a, b, x) == np.array([
        [20, 25, 34],
        [39, 48, 60],
        [68, 79, 95],
    ])).all()


def test_ndcerp_three_dimensions():
    '''The interpolation should still work with three dimensional
    arrays.
    '''
    size = (3, 3, 3)
    dims = len(size)
    ab_shape = tuple([2 ** dims // 2, *size])
    length = prod(ab_shape)
    a = np.arange(length).reshape(ab_shape)
    b = a ** 2
    x = np.full((dims, *size), .5)
    assert (lp.n_dimensional_cubic_interpolation(a, b, x) == np.array([
        [
            [1282, 1324, 1382],
            [1408, 1455, 1516],
            [1544, 1594, 1659],
        ],
        [
            [1688, 1742, 1811],
            [1843, 1898, 1972],
            [2005, 2064, 2142],
        ],
        [
            [2177, 2239, 2321],
            [2358, 2423, 2509],
            [2548, 2616, 2706],
        ],
    ])).all()


# Tests for n_dimensional_linear_interpolation.
def test_ndlerp_two_dimensions():
    '''Given two arrays of values and an array of distances,
    :funct:`n_dimensional_linear_interpolation` should return
    an array with the bilinear interpolation of the value arrays.
    '''
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
    ])
    b = np.array([
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
    x = np.array([
        [
            [0.5, 0.5, 0.5, ],
            [0.5, 0.5, 0.5, ],
            [0.5, 0.5, 0.5, ],
        ],
        [
            [0.5, 0.5, 0.5, ],
            [0.5, 0.5, 0.5, ],
            [0.5, 0.5, 0.5, ],
        ],
    ])
    assert (lp.n_dimensional_cubic_interpolation(a, b, x) == np.array([
        [0.87109375, 2.0, 3.12890625],
        [1.87109375, 3.0, 4.12890625],
        [2.87109375, 4.0, 5.12890625],
    ])).all()
