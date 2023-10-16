"""
test_ops
~~~~~~~~

Unit tests for the :mod:`pjimg.blends.ops` module.
"""
import numpy as np
import pytest as pt

from pjimg.blends import ops as blends


# Fixtures.
@pt.fixture
def a():
    """A :class:`numpy.ndarray` of image data for testing."""
    yield np.array([
        [
            [0.00, 0.25, 0.50, 0.75, 1.00,],
            [0.25, 0.50, 0.75, 1.00, 0.75,],
            [0.50, 0.75, 1.00, 0.75, 0.50,],
            [0.75, 1.00, 0.75, 0.50, 0.25,],
            [1.00, 0.75, 0.50, 0.25, 0.00,],
        ],
    ], dtype=float)


@pt.fixture
def b():
    """A :class:`numpy.ndarray` of images data for testing."""
    yield np.array([
        [
            [1.00, 0.75, 0.50, 0.25, 0.00,],
            [0.75, 1.00, 0.75, 0.50, 0.25,],
            [0.50, 0.75, 1.00, 0.75, 0.50,],
            [0.25, 0.50, 0.75, 1.00, 0.75,],
            [0.00, 0.25, 0.50, 0.75, 1.00,],
        ],
    ], dtype=float)


@pt.fixture
def c():
    """A :class:`numpy.ndarray` of images data for testing."""
    yield np.array([
        [
            [0.5000, 0.3750, 0.2500, 0.1250, 0.0000,],
            [0.3750, 0.2500, 0.1250, 0.0000, 0.1250,],
            [0.2500, 0.1250, 0.0000, 0.1250, 0.2500,],
            [0.1250, 0.0000, 0.1250, 0.2500, 0.3750,],
            [0.0000, 0.1250, 0.2500, 0.3750, 0.5000,],
        ],
    ], dtype=float)


@pt.fixture
def d():
    """A :class:`numpy.ndarray` of images data for testing."""
    yield np.array([
        [
            [0.0000, 0.1250, 0.2500, 0.3750, 0.5000,],
            [0.1250, 0.0000, 0.1250, 0.2500, 0.3750,],
            [0.2500, 0.1250, 0.0000, 0.1250, 0.2500,],
            [0.3750, 0.2500, 0.1250, 0.0000, 0.1250,],
            [0.5000, 0.3750, 0.2500, 0.1250, 0.0000,],
        ],
    ], dtype=float)


# Test cases.
def test_color_burn(a, b):
    """When blending image data, :func:`color_burn` should divide the
    value in the base image by the value in the blending image.
    """
    result = blends.color_burn(a, b)
    assert (np.around(result, 4) == np.array([
        [
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.5000, 0.6667, 1.0000, 0.0000],
            [0.0000, 0.6667, 1.0000, 0.6667, 0.0000],
            [0.0000, 1.0000, 0.6667, 0.5000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        ],
    ], dtype=float)).all()


def test_color_dodge(c, d):
    """When blending image data, :func:`color_burn` should increase the
    value in the base image by an amount relative to the value in the
    blending image. The relation is through dividing by the inverse of
    the blending image.
    """
    result = blends.color_dodge(c, d)
    assert (np.around(result, 4) == np.array([
        [
            [0.5000, 0.4286, 0.3333, 0.2000, 0.0000],
            [0.4286, 0.2500, 0.1429, 0.0000, 0.2000],
            [0.3333, 0.1429, 0.0000, 0.1429, 0.3333],
            [0.2000, 0.0000, 0.1429, 0.2500, 0.4286],
            [0.0000, 0.2000, 0.3333, 0.4286, 0.5000],
        ],
    ], dtype=float)).all()


def test_darker(a, b):
    """When blending image data, :func:`darker` should always
    take the lowest value.
    """
    result = blends.darker(a, b)
    assert (np.around(result, 4) == np.array([
        [
            [0.00, 0.25, 0.50, 0.25, 0.00,],
            [0.25, 0.50, 0.75, 0.50, 0.25,],
            [0.50, 0.75, 1.00, 0.75, 0.50,],
            [0.25, 0.50, 0.75, 0.50, 0.25,],
            [0.00, 0.25, 0.50, 0.25, 0.00,],
        ],
    ], dtype=float)).all()


def test_difference(a, b):
    """When blending image data, :func:`difference` should take the
    absolute value of the difference between the two colors.
    """
    result = blends.difference(a, b)
    assert (np.around(result, 4) == np.array([
        [
            [1.0000, 0.5000, 0.0000, 0.5000, 1.0000],
            [0.5000, 0.5000, 0.0000, 0.5000, 0.5000],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.5000, 0.5000, 0.0000, 0.5000, 0.5000],
            [1.0000, 0.5000, 0.0000, 0.5000, 1.0000],
        ],
    ], dtype=float)).all()


def test_exclusion(a, b):
    """When blending image data, :func:`exclusion` should subtract the
    double product of the colors from the sum of the colors.
    """
    result = blends.exclusion(a, b)
    assert (np.around(result, 4) == np.array([
        [
            [1.0000, 0.6250, 0.5000, 0.6250, 1.0000],
            [0.6250, 0.5000, 0.3750, 0.5000, 0.6250],
            [0.5000, 0.3750, 0.0000, 0.3750, 0.5000],
            [0.6250, 0.5000, 0.3750, 0.5000, 0.6250],
            [1.0000, 0.6250, 0.5000, 0.6250, 1.0000],
        ],
    ], dtype=float)).all()


def test_hard_light(a, b):
    """When blending image data, :func:`hard_light` should perform a
    hard light blend.
    """
    result = blends.hard_light(a, b)
    assert (np.around(result, 4) == np.array([
        [
            [0.0000, 0.3750, 0.5000, 0.6250, 1.0000],
            [0.3750, 1.0000, 0.8750, 1.0000, 0.6250],
            [0.5000, 0.8750, 1.0000, 0.8750, 0.5000],
            [0.6250, 1.0000, 0.8750, 1.0000, 0.3750],
            [1.0000, 0.6250, 0.5000, 0.3750, 0.0000],
        ],
    ], dtype=float)).all()


def test_hard_mix(a, b):
    """When blending image data, :func:`hard_mix` should perform a
    hard mix blend.
    """
    result = blends.hard_mix(a, b)
    assert (np.around(result, 4) == np.array([
        [
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 1.0000, 1.0000, 1.0000, 0.0000],
            [0.0000, 1.0000, 1.0000, 1.0000, 0.0000],
            [0.0000, 1.0000, 1.0000, 1.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        ],
    ], dtype=float)).all()


def test_lighter(a, b):
    """When blending image data, :func:`lighter` should always take the
    highest value.
    """
    result = blends.lighter(a, b)
    assert (np.around(result, 4) == np.array([
        [
            [1.0000, 0.7500, 0.5000, 0.7500, 1.0000],
            [0.7500, 1.0000, 0.7500, 1.0000, 0.7500],
            [0.5000, 0.7500, 1.0000, 0.7500, 0.5000],
            [0.7500, 1.0000, 0.7500, 1.0000, 0.7500],
            [1.0000, 0.7500, 0.5000, 0.7500, 1.0000],
        ],
    ], dtype=float)).all()


def test_linear_burn(a, b):
    """When blending image data, :func:`linear_burn` should divide the
    value in the base image by the value in the blending image.
    """
    result = blends.linear_burn(a, b)
    assert (np.around(result, 4) == np.array([
        [
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.5000, 0.5000, 0.5000, 0.0000],
            [0.0000, 0.5000, 1.0000, 0.5000, 0.0000],
            [0.0000, 0.5000, 0.5000, 0.5000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        ],
    ], dtype=float)).all()


def test_linear_dodge(c, d):
    """When blending image data, :func:`linear_dodge` should add the
    colors together.
    """
    result = blends.linear_dodge(c, d)
    assert (np.around(result, 4) == np.array([
        [
            [0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
            [0.5000, 0.2500, 0.2500, 0.2500, 0.5000],
            [0.5000, 0.2500, 0.0000, 0.2500, 0.5000],
            [0.5000, 0.2500, 0.2500, 0.2500, 0.5000],
            [0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
        ],
    ], dtype=float)).all()


def test_linear_light(a, b):
    """When blending image data, :func:`linear_light` should add the
    colors together.
    """
    result = blends.linear_light(a, b)
    assert (np.around(result, 4) == np.array([
        [
            [0.0000, 0.2500, 0.5000, 0.7500, 1.0000],
            [0.2500, 1.0000, 1.0000, 1.0000, 0.7500],
            [0.5000, 1.0000, 1.0000, 1.0000, 0.5000],
            [0.7500, 1.0000, 1.0000, 1.0000, 0.2500],
            [1.0000, 0.7500, 0.5000, 0.2500, 0.0000],
        ],
    ], dtype=float)).all()


def test_multiply(a, b):
    """When blending image data, :func:`multiply` should multiply the
    two values.
    """
    result = blends.multiply(a, b)
    assert (np.around(result, 4) == np.array([
        [
            [0.0000, 0.1875, 0.2500, 0.1875, 0.0000, ],
            [0.1875, 0.5000, 0.5625, 0.5000, 0.1875, ],
            [0.2500, 0.5625, 1.0000, 0.5625, 0.2500, ],
            [0.1875, 0.5000, 0.5625, 0.5000, 0.1875, ],
            [0.0000, 0.1875, 0.2500, 0.1875, 0.0000, ],
        ],
    ], dtype=float)).all()


def test_overlay(a, b):
    """When blending image data, :func:`overlay` should perform an
    overlay blend.
    """
    result = blends.overlay(a, b)
    assert (np.around(result, 4) == np.array([
        [
            [0.0000, 0.3750, 0.5000, 0.6250, 1.0000],
            [0.3750, 1.0000, 0.8750, 1.0000, 0.6250],
            [0.5000, 0.8750, 1.0000, 0.8750, 0.5000],
            [0.6250, 1.0000, 0.8750, 1.0000, 0.3750],
            [1.0000, 0.6250, 0.5000, 0.3750, 0.0000],
        ],
    ], dtype=float)).all()


def test_pin_light(a, b):
    """When blending image data, :func:`pin_light` should perform an pin
    light blend.
    """
    result = blends.pin_light(a, b)
    assert (np.around(result, 4) == np.array([
        [
            [0.0000, 0.5000, 0.5000, 0.5000, 1.0000],
            [0.5000, 1.0000, 0.7500, 1.0000, 0.5000],
            [0.5000, 0.7500, 1.0000, 0.7500, 0.5000],
            [0.5000, 1.0000, 0.7500, 1.0000, 0.5000],
            [1.0000, 0.5000, 0.5000, 0.5000, 0.0000],
        ],
    ], dtype=float)).all()


def test_replace(a, b):
    """When blending image data, :func:`replace` should return the
    second set.
    """
    result = blends.replace(a, b)
    assert (np.around(result, 4) == b).all()


def test_screen(a, b):
    """When blending image data, :func:`screen` should increase the
    value in the base image by an amount relative to the value in the
    blending image.
    """
    result = blends.screen(a, b)
    assert (np.around(result, 4) == np.array([
        [
            [1.0000, 0.8125, 0.7500, 0.8125, 1.0000],
            [0.8125, 1.0000, 0.9375, 1.0000, 0.8125],
            [0.7500, 0.9375, 1.0000, 0.9375, 0.7500],
            [0.8125, 1.0000, 0.9375, 1.0000, 0.8125],
            [1.0000, 0.8125, 0.7500, 0.8125, 1.0000],
        ],
    ], dtype=float)).all()


def test_soft_light(a, b):
    """When blending image data, :func:`soft_light` should perform an
    soft light blend.
    """
    result = blends.soft_light(a, b)
    assert (np.around(result, 4) == np.array([
        [
            [1.0000, 0.6562, 0.5000, 0.3750, 0.0000],
            [0.6562, 1.0000, 0.8080, 0.7071, 0.3750],
            [0.5000, 0.8080, 1.0000, 0.8080, 0.5000],
            [0.3750, 0.7071, 0.8080, 1.0000, 0.6562],
            [0.0000, 0.3750, 0.5000, 0.6562, 1.0000],
        ],
    ], dtype=float)).all()


def test_vivid_light(a, b):
    """When blending image data, :func:`vivid_light` should perform an
    vivid light blend.
    """
    result = blends.vivid_light(a, b)
    assert (np.around(result, 4) == np.array([
        [
            [0.0000, 0.5000, 0.5000, 0.5000, 0.0000],
            [0.5000, 1.0000, 1.0000, 0.0000, 0.5000],
            [0.5000, 1.0000, 0.0000, 1.0000, 0.5000],
            [0.5000, 0.0000, 1.0000, 1.0000, 0.5000],
            [0.0000, 0.5000, 0.5000, 0.5000, 0.0000],
        ],
    ], dtype=float)).all()
