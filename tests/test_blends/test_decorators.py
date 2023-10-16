"""
test_decorators
~~~~~~~~~~~~~~~

Unit tests for :mod:`pjimg.blends.decorator`.
"""
import numpy as np
import pytest as pt

from pjimg.blends import decorators as c


# Fixtures.
@pt.fixture
def a():
    """An array of zeros for testing."""
    yield np.zeros((1, 5, 5), dtype=float)


@pt.fixture
def b():
    """An array of ones for testing."""
    yield np.ones((1, 5, 5), dtype=float)


# Test cases.
class TestCanFade:
    def test_fades(self, a, b):
        """When given a fade amount, :func:`can_fade` should
        adjust how much the blending operation affects the base
        image by the given amount.
        """
        @c.can_fade
        def spam(a, b):
            return b

        result = spam(a, b, 0.5)
        assert (np.around(result, 4) == np.array([
            [
                [0.5, 0.5, 0.5, 0.5, 0.5,],
                [0.5, 0.5, 0.5, 0.5, 0.5,],
                [0.5, 0.5, 0.5, 0.5, 0.5,],
                [0.5, 0.5, 0.5, 0.5, 0.5,],
                [0.5, 0.5, 0.5, 0.5, 0.5,],
            ],
        ], dtype=float)).all()

    def test_no_fades(self, a, b):
        """If no fade is passed, :func:`can_fade` should not change the
        returned data.
        """
        @c.can_fade
        def spam(a, b):
            return b

        result = spam(a, b)
        assert (np.around(result, 4) == b).all()


class TestCanMask:
    def test_mask(self, a, b):
        """When applied to a function, :func:`can_mask` should
        adjust how much the blending operation affects each value
        of the base image based on the appropriate value of the given
        mask.
        """
        @c.can_mask
        def spam(a, b):
            return b

        mask = np.array([
            [
                [1.00, 1.00, 1.00, 1.00, 1.00,],
                [0.75, 0.75, 0.75, 0.75, 0.75,],
                [0.50, 0.50, 0.50, 0.50, 0.50,],
                [0.25, 0.25, 0.25, 0.25, 0.25,],
                [0.00, 0.00, 0.00, 0.00, 0.00,],
            ],
        ], dtype=float)
        result = spam(b, a, mask)
        assert (np.around(result, 4) == np.array([
            [
                [0.00, 0.00, 0.00, 0.00, 0.00,],
                [0.25, 0.25, 0.25, 0.25, 0.25,],
                [0.50, 0.50, 0.50, 0.50, 0.50,],
                [0.75, 0.75, 0.75, 0.75, 0.75,],
                [1.00, 1.00, 1.00, 1.00, 1.00,],
            ],
        ], dtype=float)).all()

    def test_no_mask(self, a, b):
        """If no mask is passed, :func:`can_mask` should not change the
        returned data.
        """
        @c.can_mask
        def spam(a, b):
            return b

        result = spam(b, a)
        assert (np.around(result, 4) == a).all()


class TestWillClip:
    def test_clips(self):
        """When applied to a function, :func:`will_clip` should
        set any values in the output of the decorated function that
        are greater than one to one and any values that are less
        than zero to zero.
        """
        @c.will_clip
        def spam(a, b):
            return a + b

        a = np.array([
            [
                [-0.5, 0.0, 0.5, 1.0, 1.5,],
                [-0.5, 0.0, 0.5, 1.0, 1.5,],
                [-0.5, 0.0, 0.5, 1.0, 1.5,],
                [-0.5, 0.0, 0.5, 1.0, 1.5,],
                [-0.5, 0.0, 0.5, 1.0, 1.5,],
            ],
        ], dtype=float)
        b = np.array([
            [
                [0.0, 0.0, 0.0, 0.0, 0.0,],
                [0.0, 0.0, 0.0, 0.0, 0.0,],
                [0.0, 0.0, 0.0, 0.0, 0.0,],
                [0.0, 0.0, 0.0, 0.0, 0.0,],
                [0.0, 0.0, 0.0, 0.0, 0.0,],
            ],
        ], dtype=float)
        result = spam(a, b)
        assert (np.around(result, 4) == np.array([
            [
                [0.0, 0.0, 0.5, 1.0, 1.0,],
                [0.0, 0.0, 0.5, 1.0, 1.0,],
                [0.0, 0.0, 0.5, 1.0, 1.0,],
                [0.0, 0.0, 0.5, 1.0, 1.0,],
                [0.0, 0.0, 0.5, 1.0, 1.0,],
            ]
        ], dtype=float)).all()


class TestWillColorize:
    def test_colorize_a(self):
        """Given an grayscale image and a RGB image, :func:`will_colorize`
        should convert the grayscale to RGB.
        """
        @c.will_colorize
        def spam(a, b):
            return a

        a = np.zeros((1, 5, 5), dtype=float)
        b = np.ones((1, 5, 5, 3), dtype=float)
        result = spam(a, b)
        assert (np.around(result, 4) == np.zeros(
            (1, 5, 5, 3), dtype=float
        )).all()

    def test_colorize_b(self):
        """Given an RGB image and a grayscale image, :func:`will_colorize`
        should convert the grayscale to RGB.
        """
        @c.will_colorize
        def spam(a, b):
            return b

        a = np.zeros((1, 5, 5, 3), dtype=float)
        b = np.ones((1, 5, 5), dtype=float)
        result = spam(a, b)
        assert (np.around(result, 4) == np.ones(
            (1, 5, 5, 3), dtype=float
        )).all()

    def test_no_effect_when_both_grayscale(self):
        """If both images only have one channel, :func:`will_colorize`
        shouldn't change either array.
        """
        @c.will_colorize
        def spam(a, b):
            return a + b

        a = np.zeros((1, 5, 5), dtype=float)
        b = np.ones((1, 5, 5), dtype=float)
        result = spam(a, b)
        assert (np.around(result, 4) == b).all()

    def test_no_effect_when_both_rgb(self):
        """If both images have three channels, :func:`will_colorize`
        shouldn't change either array.
        """
        @c.will_colorize
        def spam(a, b):
            return a + b

        a = np.zeros((1, 5, 5, 3), dtype=float)
        b = np.ones((1, 5, 5, 3), dtype=float)
        result = spam(a, b)
        assert (np.around(result, 4) == b).all()

    def test_no_effect_when_off(self):
        """If colorize is given `False`, :func:`will_colorize`
        shouldn't change either array.
        """
        @c.will_colorize
        def spam(a, b):
            return b

        a = np.zeros((1, 5, 5, 3), dtype=float)
        b = np.ones((1, 5, 5), dtype=float)
        result = spam(a, b, colorize=False)
        assert (np.around(result, 4) == b).all()


class TestWillMatchSize:
    def test_clips(self):
        """When applied to a function, the will_match_size decorator
        should increase the size of a smaller image to the size of
        the larger image. The fill for the added area should be black.
        """
        @c.will_match_size
        def spam(a, b):
            return a + b

        a = np.array([
            [
                [0.0, 0.0, 0.5, 1.0, 1.0,],
                [0.0, 0.0, 0.5, 1.0, 1.0,],
                [0.0, 0.0, 0.5, 1.0, 1.0,],
                [0.0, 0.0, 0.5, 1.0, 1.0,],
                [0.0, 0.0, 0.5, 1.0, 1.0,],
            ],
        ], dtype=float)
        b = np.array([
            [
                [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5,],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
                [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5,],
            ],
        ], dtype=float)
        result = spam(a, b)
        assert (np.around(result, 4) == np.array([
            [
                [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5,],
                [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 0.0,],
                [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 0.0,],
                [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 0.0,],
                [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 0.0,],
                [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 0.0,],
                [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5,],
            ]
        ], dtype=float)).all()
