"""
test_eases
~~~~~~~~~~

Unit tests for :mod:`pjimg.eases`.
"""
import numpy as np
import pytest as pt

import pjimg.eases.eases_ as ie


# Fixtures.
@pt.fixture
def a():
    """A sample :class:`numpy.ndarray` for testing."""
    yield np.array([
        [
            [0.00, 0.25, 0.50, 0.75, 1.00, ],
            [0.25, 0.50, 0.75, 1.00, 0.75, ],
            [0.50, 0.75, 1.00, 0.75, 0.50, ],
            [0.75, 1.00, 0.75, 0.50, 0.25, ],
            [1.00, 0.75, 0.50, 0.25, 0.00, ],
        ],
    ], dtype=float)


@pt.fixture
def e():
    """A sample :class:`numpy.ndarray` for testing."""
    yield np.array([
        [
            [0.0, 0.1, 0.2, 0.3, 0.4, ],
            [0.2, 0.3, 0.4, 0.5, 0.6, ],
            [0.4, 0.5, 0.6, 0.7, 0.8, ],
            [0.6, 0.7, 0.8, 0.9, 1.0, ],
            [0.8, 0.9, 1.0, 0.7, 0.8, ],
        ],
    ], dtype=float)


# Tests for ease in functions.
def test_in_back(a):
    """Given an array of image data, :func:`in_back` should run
    the 'in back' easing function on the data and return the result.
    """
    result = ie.in_back(a)
    assert (np.around(result, 4) == np.array([
        [
            [0.0000, -0.0641, -0.0877, 0.1826, 1.0000],
            [-0.0641, -0.0877, 0.1826, 1.0000, 0.1826],
            [-0.0877, 0.1826, 1.0000, 0.1826, -0.0877],
            [0.1826, 1.0000, 0.1826, -0.0877, -0.0641],
            [1.0000, 0.1826, -0.0877, -0.0641, 0.0000],
        ],
    ], dtype=float)).all()


def test_in_circ(a):
    """Given an array of image data, :func:`in_circ` should run
    the 'in circ' easing function on the data and return the result.
    """
    result = ie.in_circ(a)
    assert (np.around(result, 4) == np.array([
        [
            [0.0000, 0.0318, 0.1340, 0.3386, 1.0000],
            [0.0318, 0.1340, 0.3386, 1.0000, 0.3386],
            [0.1340, 0.3386, 1.0000, 0.3386, 0.1340],
            [0.3386, 1.0000, 0.3386, 0.1340, 0.0318],
            [1.0000, 0.3386, 0.1340, 0.0318, 0.0000],
        ],
    ], dtype=float)).all()


def test_in_cubic(a):
    """Given an array of image data, :func:`in_cubic` should run
    the 'in cubic' easing function on the data and return the result.
    """
    result = ie.in_cubic(a)
    assert (np.around(result, 4) == np.array([
        [
            [0.0000, 0.0156, 0.1250, 0.4219, 1.0000],
            [0.0156, 0.1250, 0.4219, 1.0000, 0.4219],
            [0.1250, 0.4219, 1.0000, 0.4219, 0.1250],
            [0.4219, 1.0000, 0.4219, 0.1250, 0.0156],
            [1.0000, 0.4219, 0.1250, 0.0156, 0.0000],
        ],
    ], dtype=float)).all()


def test_in_elastic(a):
    """Given an array of image data, :func:`in_elastic` should run
    the 'in elastic' easing function on the data and return the result.
    """
    result = ie.in_elastic(a)
    assert (np.around(result, 4) == np.array([
        [
            [0.0000, -0.0055, -0.0156, 0.0884, 1.0000],
            [-0.0055, -0.0156, 0.0884, 1.0000, 0.0884],
            [-0.0156, 0.0884, 1.0000, 0.0884, -0.0156],
            [0.0884, 1.0000, 0.0884, -0.0156, -0.0055],
            [1.0000, 0.0884, -0.0156, -0.0055, 0.0000],
        ],
    ], dtype=float)).all()


def test_in_quad(a):
    """Given an array of image data, :func:`in_quad` should run
    the 'in quad' easing function on the data and return the result.
    """
    result = ie.in_quad(a)
    assert (np.around(result, 4) == np.array([
        [
            [0.0000, 0.0625, 0.2500, 0.5625, 1.0000],
            [0.0625, 0.2500, 0.5625, 1.0000, 0.5625],
            [0.2500, 0.5625, 1.0000, 0.5625, 0.2500],
            [0.5625, 1.0000, 0.5625, 0.2500, 0.0625],
            [1.0000, 0.5625, 0.2500, 0.0625, 0.0000],
        ],
    ], dtype=float)).all()


def test_in_quint(a):
    """Given an array of image data, :func:`in_quint` should run
    the 'in quad' easing function on the data and return the result.
    """
    result = ie.in_quint(a)
    assert (np.around(result, 4) == np.array([
        [
            [0.0000, 0.0010, 0.0312, 0.2373, 1.0000],
            [0.0010, 0.0312, 0.2373, 1.0000, 0.2373],
            [0.0312, 0.2373, 1.0000, 0.2373, 0.0312],
            [0.2373, 1.0000, 0.2373, 0.0312, 0.0010],
            [1.0000, 0.2373, 0.0312, 0.0010, 0.0000],
        ],
    ], dtype=float)).all()


def test_in_sin(a):
    """Given an array of image data, :func:`in_sin` should run
    the 'in sin' easing function on the data and return the result.
    """
    result = ie.in_sin(a)
    assert (np.around(result, 4) == np.array([
        [
            [0.0000, 0.0761, 0.2929, 0.6173, 1.0000],
            [0.0761, 0.2929, 0.6173, 1.0000, 0.6173],
            [0.2929, 0.6173, 1.0000, 0.6173, 0.2929],
            [0.6173, 1.0000, 0.6173, 0.2929, 0.0761],
            [1.0000, 0.6173, 0.2929, 0.0761, 0.0000],
        ],
    ], dtype=float)).all()


# Tests for ease in out functions.
def test_in_out_back(a):
    """Given an array of image data, :func:`in_out_back` should
    run the 'in out back' easing function on the data and return the
    result.
    """
    result = ie.in_out_back(a)
    assert (np.around(result, 4) == np.array([
        [
            [-0.0000, -0.0997, 0.5000, 1.0997, 1.0000],
            [-0.0997, 0.5000, 1.0997, 1.0000, 1.0997],
            [0.5000, 1.0997, 1.0000, 1.0997, 0.5000],
            [1.0997, 1.0000, 1.0997, 0.5000, -0.0997],
            [1.0000, 1.0997, 0.5000, -0.0997, -0.0000],
        ],
    ], dtype=float)).all()


def test_in_out_circ(a):
    """Given an array of image data, :func:`in_out_circ` should
    run the 'in out circ' easing function on the data and return the
    result.
    """
    result = ie.in_out_circ(a)
    assert (np.around(result, 4) == np.array([
        [
            [0.0000, 0.0670, 0.5000, 0.9330, 1.0000],
            [0.0670, 0.5000, 0.9330, 1.0000, 0.9330],
            [0.5000, 0.9330, 1.0000, 0.9330, 0.5000],
            [0.9330, 1.0000, 0.9330, 0.5000, 0.0670],
            [1.0000, 0.9330, 0.5000, 0.0670, 0.0000],
        ],
    ], dtype=float)).all()


def test_in_out_cos(a):
    """Given an array of image data, :func:`in_out_cos` should
    run the 'in out cos' easing function on the data and return the
    result.
    """
    result = ie.in_out_cos(a)
    assert (np.around(result, 4) == np.array([
        [
            [0.5000, 0.1464, -0.0000, 0.1464, 0.5000],
            [0.1464, -0.0000, 0.1464, 0.5000, 0.1464],
            [-0.0000, 0.1464, 0.5000, 0.1464, -0.0000],
            [0.1464, 0.5000, 0.1464, -0.0000, 0.1464],
            [0.5000, 0.1464, -0.0000, 0.1464, 0.5000],
        ],
    ], dtype=float)).all()


def test_in_out_cubic(a):
    """Given an array of image data, :func:`in_out_cubic` should
    run the 'in out cubic' easing function on the data and return the
    result.
    """
    result = ie.in_out_cubic(a)
    assert (np.around(result, 4) == np.array([
        [
            [0.0000, 0.0625, 0.5000, 0.9375, 1.0000],
            [0.0625, 0.5000, 0.9375, 1.0000, 0.9375],
            [0.5000, 0.9375, 1.0000, 0.9375, 0.5000],
            [0.9375, 1.0000, 0.9375, 0.5000, 0.0625],
            [1.0000, 0.9375, 0.5000, 0.0625, 0.0000],
        ],
    ], dtype=float)).all()


def test_in_out_elastic(a):
    """Given an array of image data, :func:`in_out_elastic` should
    run the 'in out elastic' easing function on the data and return the
    result.
    """
    result = ie.in_out_elastic(a)
    assert (np.around(result, 4) == np.array([
        [
            [0.0000, 0.0120, 0.5000, 0.9880, 1.0000],
            [0.0120, 0.5000, 0.9880, 1.0000, 0.9880],
            [0.5000, 0.9880, 1.0000, 0.9880, 0.5000],
            [0.9880, 1.0000, 0.9880, 0.5000, 0.0120],
            [1.0000, 0.9880, 0.5000, 0.0120, 0.0000],
        ],
    ], dtype=float)).all()


def test_in_out_quad(a):
    """Given an array of image data, :func:`in_out_quad` should
    run the 'in out quad' easing function on the data and return the
    result.
    """
    result = ie.in_out_quad(a)
    assert (np.around(result, 4) == np.array([
        [
            [0.0000, 0.1250, 0.5000, 0.8750, 1.0000],
            [0.1250, 0.5000, 0.8750, 1.0000, 0.8750],
            [0.5000, 0.8750, 1.0000, 0.8750, 0.5000],
            [0.8750, 1.0000, 0.8750, 0.5000, 0.1250],
            [1.0000, 0.8750, 0.5000, 0.1250, 0.0000],
        ],
    ], dtype=float)).all()


def test_in_out_quint(a):
    """Given an array of image data, :func:`in_out_quint` should
    run the 'in out quint' easing function on the data and return the
    result.
    """
    result = ie.in_out_quint(a)
    assert (np.around(result, 4) == np.array([
        [
            [0.0000, 0.0156, 0.5000, 0.9844, 1.0000],
            [0.0156, 0.5000, 0.9844, 1.0000, 0.9844],
            [0.5000, 0.9844, 1.0000, 0.9844, 0.5000],
            [0.9844, 1.0000, 0.9844, 0.5000, 0.0156],
            [1.0000, 0.9844, 0.5000, 0.0156, 0.0000],
        ],
    ], dtype=float)).all()


def test_in_out_sin(a):
    """Given an array of image data, :func:`in_out_sin` should
    run the 'in out sin' easing function on the data and return the
    result.
    """
    result = ie.in_out_sin(a)
    assert (np.around(result, 4) == np.array([
        [
            [0.0000, 0.1464, 0.5000, 0.8536, 1.0000],
            [0.1464, 0.5000, 0.8536, 1.0000, 0.8536],
            [0.5000, 0.8536, 1.0000, 0.8536, 0.5000],
            [0.8536, 1.0000, 0.8536, 0.5000, 0.1464],
            [1.0000, 0.8536, 0.5000, 0.1464, 0.0000],
        ],
    ], dtype=float)).all()


# Tests for ease mid.
def test_mid_bump_linear(e):
    """Given an array of image data, :func:`mid_bump_linear` should
    run the 'mid bump linear' easing function on the data and return the
    result.
    """
    result = ie.mid_bump_linear(e)
    assert (np.around(result, 4) == np.array([
        [
            [0.0000, 0.0000, 0.0000, 0.2000, 0.6000],
            [0.0000, 0.2000, 0.6000, 1.0000, 0.6000],
            [0.6000, 1.0000, 0.6000, 0.2000, 0.0000],
            [0.6000, 0.2000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.2000, 0.0000],
        ],
    ], dtype=float)).all()


def test_mid_bump_sin(e):
    """Given an array of image data, :func:`mid_bump_sin` should
    run the 'mid bump sin' easing function on the data and return the
    result.
    """
    result = ie.mid_bump_sin(e)
    assert (np.around(result, 4) == np.array([
        [
            [0.0000, 0.0000, 0.0000, 0.0955, 0.6545],
            [0.0000, 0.0955, 0.6545, 1.0000, 0.6545],
            [0.6545, 1.0000, 0.6545, 0.0955, 0.0000],
            [0.6545, 0.0955, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0955, 0.0000],
        ],
    ], dtype=float)).all()


# Tests for ease out.
def test_out_bounce(a):
    """Given an array of image data, :func:`out_bounce` should
    run the 'out bounce' easing function on the data and return the
    result.
    """
    result = ie.out_bounce(a)
    assert (np.around(result, 4) == np.array([
        [
            [0.0000, 0.4727, 0.7656, 0.9727, 1.0000],
            [0.4727, 0.7656, 0.9727, 1.0000, 0.9727],
            [0.7656, 0.9727, 1.0000, 0.9727, 0.7656],
            [0.9727, 1.0000, 0.9727, 0.7656, 0.4727],
            [1.0000, 0.9727, 0.7656, 0.4727, 0.0000],
        ],
    ], dtype=float)).all()


def test_out_circ(a):
    """Given an array of image data, :func:`out_circ` should
    run the 'out circ' easing function on the data and return the
    result.
    """
    result = ie.out_circ(a)
    assert (np.around(result, 4) == np.array([
        [
            [0.0000, 0.6614, 0.8660, 0.9682, 1.0000],
            [0.6614, 0.8660, 0.9682, 1.0000, 0.9682],
            [0.8660, 0.9682, 1.0000, 0.9682, 0.8660],
            [0.9682, 1.0000, 0.9682, 0.8660, 0.6614],
            [1.0000, 0.9682, 0.8660, 0.6614, 0.0000],
        ],
    ], dtype=float)).all()


def test_out_cubic(a):
    """Given an array of image data, :func:`out_cubic` should
    run the 'out cubic' easing function on the data and return the
    result.
    """
    result = ie.out_cubic(a)
    assert (np.around(result, 4) == np.array([
        [
            [0.0000, 0.5781, 0.8750, 0.9844, 1.0000],
            [0.5781, 0.8750, 0.9844, 1.0000, 0.9844],
            [0.8750, 0.9844, 1.0000, 0.9844, 0.8750],
            [0.9844, 1.0000, 0.9844, 0.8750, 0.5781],
            [1.0000, 0.9844, 0.8750, 0.5781, 0.0000],
        ],
    ], dtype=float)).all()


def test_out_elastic(a):
    """Given an array of image data, :func:`out_elastic` should
    run the 'out elastic' easing function on the data and return the
    result.
    """
    result = ie.out_elastic(a)
    assert (np.around(result, 4) == np.array([
        [
            [0.0000, 0.9116, 1.0156, 1.0055, 1.0000],
            [0.9116, 1.0156, 1.0055, 1.0000, 1.0055],
            [1.0156, 1.0055, 1.0000, 1.0055, 1.0156],
            [1.0055, 1.0000, 1.0055, 1.0156, 0.9116],
            [1.0000, 1.0055, 1.0156, 0.9116, 0.0000],
        ],
    ], dtype=float)).all()


def test_out_quad(a):
    """Given an array of image data, :func:`out_quad` should
    run the 'out quad' easing function on the data and return the
    result.
    """
    result = ie.out_quad(a)
    assert (np.around(result, 4) == np.array([
        [
            [0.0000, 0.4375, 0.7500, 0.9375, 1.0000],
            [0.4375, 0.7500, 0.9375, 1.0000, 0.9375],
            [0.7500, 0.9375, 1.0000, 0.9375, 0.7500],
            [0.9375, 1.0000, 0.9375, 0.7500, 0.4375],
            [1.0000, 0.9375, 0.7500, 0.4375, 0.0000],
        ],
    ], dtype=float)).all()


def test_out_quint(a):
    """Given an array of image data, :func:`out_quint` should
    run the 'out quint' easing function on the data and return the
    result.
    """
    result = ie.out_quint(a)
    assert (np.around(result, 4) == np.array([
        [
            [0.0000, 0.7627, 0.9688, 0.9990, 1.0000],
            [0.7627, 0.9688, 0.9990, 1.0000, 0.9990],
            [0.9688, 0.9990, 1.0000, 0.9990, 0.9688],
            [0.9990, 1.0000, 0.9990, 0.9688, 0.7627],
            [1.0000, 0.9990, 0.9688, 0.7627, 0.0000],
        ],
    ], dtype=float)).all()


def test_out_sin(a):
    """Given an array of image data, :func:`out_sin` should
    run the 'out sin' easing function on the data and return the
    result.
    """
    result = ie.out_sin(a)
    assert (np.around(result, 4) == np.array([
        [
            [0.0000, 0.3827, 0.7071, 0.9239, 1.0000],
            [0.3827, 0.7071, 0.9239, 1.0000, 0.9239],
            [0.7071, 0.9239, 1.0000, 0.9239, 0.7071],
            [0.9239, 1.0000, 0.9239, 0.7071, 0.3827],
            [1.0000, 0.9239, 0.7071, 0.3827, 0.0000],
        ],
    ], dtype=float)).all()
