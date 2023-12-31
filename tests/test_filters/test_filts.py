"""
test_filts
~~~~~~~~~~

Unit tests for :mod:`pjimg.filters.filters_`.
"""
import numpy as np
import pytest as pt

from pjimg.filters import filters_ as f


# Fixtures.
@pt.fixture
def a():
    """An array for testing."""
    yield np.array([
        [0.00, 0.25, 0.50, 0.75, 1.00,],
        [0.25, 0.50, 0.75, 1.00, 0.75,],
        [0.50, 0.75, 1.00, 0.75, 0.50,],
        [0.75, 1.00, 0.75, 0.50, 0.25,],
        [1.00, 0.75, 0.50, 0.25, 0.00,],
    ], dtype=float)


@pt.fixture
def image_1_3_3():
    """An array for testing."""
    yield np.array([
        [1.0, 0.5, 0.0,],
        [0.5, 0.0, 0.5,],
        [0.0, 0.5, 1.0,],
    ], dtype=float)


@pt.fixture
def image_5_5_low_contrast():
    """An image array for testing low contrast situations."""
    yield np.array([
        [0.3, 0.4, 0.5, 0.6, 0.7, ],
        [0.3, 0.4, 0.5, 0.6, 0.7, ],
        [0.3, 0.4, 0.5, 0.6, 0.7, ],
        [0.3, 0.4, 0.5, 0.6, 0.7, ],
        [0.3, 0.4, 0.5, 0.6, 0.7, ],
    ], dtype=float)


@pt.fixture
def image_5_5_tenths():
    """An inmage data array for testing."""
    yield np.array([
        [
            [0.0, 0.1, 0.2, 0.3, 0.4, ],
            [0.2, 0.3, 0.4, 0.5, 0.6, ],
            [0.4, 0.5, 0.6, 0.7, 0.8, ],
            [0.6, 0.7, 0.8, 0.9, 1.0, ],
            [0.8, 0.9, 1.0, 0.7, 0.8, ],
        ],
    ], dtype=float)


@pt.fixture
def video_2_3_3():
    """An array of video data for testing."""
    yield np.array([
        [
            [1.0, 0.5, 0.0, ],
            [0.5, 0.0, 0.5, ],
            [0.0, 0.5, 1.0, ],
        ],
        [
            [1.0, 0.5, 0.0, ],
            [0.5, 0.0, 0.5, ],
            [0.0, 0.5, 1.0, ],
        ],
    ], dtype=float)


@pt.fixture
def video_2_5_5():
    """An array of video data for testing."""
    yield np.array([
        [
            [0.00, 0.25, 0.50, 0.75, 1.00,],
            [0.25, 0.50, 0.75, 1.00, 0.75,],
            [0.50, 0.75, 1.00, 0.75, 0.50,],
            [0.75, 1.00, 0.75, 0.50, 0.25,],
            [1.00, 0.75, 0.50, 0.25, 0.00,],
        ],
        [
            [1.00, 0.75, 0.50, 0.25, 0.00,],
            [0.75, 1.00, 0.75, 0.50, 0.25,],
            [0.50, 0.75, 1.00, 0.75, 0.50,],
            [0.25, 0.50, 0.75, 1.00, 0.75,],
            [0.00, 0.25, 0.50, 0.75, 1.00,],
        ],
    ], dtype=float)


# Test cases.
class TestFilterColorize:
    def test_filter(self, image_1_3_3):
        """Given an RGB color and grayscale image data,
        :func:`colorize` should apply the color to
        the image data.
        """
        result = f.colorize(
            image_1_3_3,
            white='hsv(350, 100%, 100%)',
            black='hsv(10, 100%, 0%)'
        )
        assert (np.around(result, 4) == np.array([
            [
                [1.0000, 0.0000, 0.1686],
                [0.4980, 0.0000, 0.0824],
                [0.0000, 0.0000, 0.0000],
            ],
            [
                [0.4980, 0.0000, 0.0824],
                [0.0000, 0.0000, 0.0000],
                [0.4980, 0.0000, 0.0824],
            ],
            [
                [0.0000, 0.0000, 0.0000],
                [0.4980, 0.0000, 0.0824],
                [1.0000, 0.0000, 0.1686],
            ],
        ], dtype=float)).all()

    def test_by_colorkey(self, image_1_3_3):
        """Given an color key and grayscale image data,
        :func:`colorize` should apply the color to
        the image data.
        """
        result = f.colorize(
            image_1_3_3,
            colorkey='s'
        )
        assert (np.around(result, 4) == np.array([
            [
                [1.0000, 0.0000, 0.1686],
                [0.4980, 0.0000, 0.0824],
                [0.0000, 0.0000, 0.0000],
            ],
            [
                [0.4980, 0.0000, 0.0824],
                [0.0000, 0.0000, 0.0000],
                [0.4980, 0.0000, 0.0824],
            ],
            [
                [0.0000, 0.0000, 0.0000],
                [0.4980, 0.0000, 0.0824],
                [1.0000, 0.0000, 0.1686],
            ],
        ], dtype=float)).all()

    def test_on_video(self, video_2_3_3):
        """Given an RGB color and grayscale image data,
        :func:`colorize` should apply the color to
        the video data.
        """
        result = f.colorize(
            video_2_3_3,
            colorkey='s'
        )
        assert (np.around(result, 4) == np.array([
            [
                [
                    [1.0000, 0.0000, 0.1686],
                    [0.4980, 0.0000, 0.0824],
                    [0.0000, 0.0000, 0.0000],
                ],
                [
                    [0.4980, 0.0000, 0.0824],
                    [0.0000, 0.0000, 0.0000],
                    [0.4980, 0.0000, 0.0824],
                ],
                [
                    [0.0000, 0.0000, 0.0000],
                    [0.4980, 0.0000, 0.0824],
                    [1.0000, 0.0000, 0.1686],
                ],
            ],
            [
                [
                    [1.0000, 0.0000, 0.1686],
                    [0.4980, 0.0000, 0.0824],
                    [0.0000, 0.0000, 0.0000],
                ],
                [
                    [0.4980, 0.0000, 0.0824],
                    [0.0000, 0.0000, 0.0000],
                    [0.4980, 0.0000, 0.0824],
                ],
                [
                    [0.0000, 0.0000, 0.0000],
                    [0.4980, 0.0000, 0.0824],
                    [1.0000, 0.0000, 0.1686],
                ],
            ],
        ], dtype=float)).all()


class TestFilterContrast:
    def test_filter(self, image_5_5_low_contrast):
        """Given image data, :func:`contrast` adjust the
        range of the data to ensure the darkest color is black
        and the lightest is white.
        """
        result = f.contrast(image_5_5_low_contrast)
        assert (np.around(result, 4) == np.array([
            [0.0000, 0.2500, 0.5000, 0.7500, 1.0000],
            [0.0000, 0.2500, 0.5000, 0.7500, 1.0000],
            [0.0000, 0.2500, 0.5000, 0.7500, 1.0000],
            [0.0000, 0.2500, 0.5000, 0.7500, 1.0000],
            [0.0000, 0.2500, 0.5000, 0.7500, 1.0000],
        ], dtype=float)).all()

    def test_black(self, image_5_5_low_contrast):
        """Given image data and a black point, :func:`contrast`
        adjust the range of the data to ensure the darkest color is the
        given black point and the lightest is white.
        """
        result = f.contrast(image_5_5_low_contrast, black=0.5)
        assert (np.around(result, 4) == np.array([
            [0.5000, 0.6250, 0.7500, 0.8750, 1.0000],
            [0.5000, 0.6250, 0.7500, 0.8750, 1.0000],
            [0.5000, 0.6250, 0.7500, 0.8750, 1.0000],
            [0.5000, 0.6250, 0.7500, 0.8750, 1.0000],
            [0.5000, 0.6250, 0.7500, 0.8750, 1.0000],
        ], dtype=float)).all()

    def test_white(self, image_5_5_low_contrast):
        """Given image data and a white point, :func:`contrast`
        adjust the range of the data to ensure the darkest color is black
        and the lightest is the given maximum.
        """
        result = f.contrast(image_5_5_low_contrast, white=0.5)
        assert (np.around(result, 4) == np.array([
            [0.0000, 0.1250, 0.2500, 0.3750, 0.5000],
            [0.0000, 0.1250, 0.2500, 0.3750, 0.5000],
            [0.0000, 0.1250, 0.2500, 0.3750, 0.5000],
            [0.0000, 0.1250, 0.2500, 0.3750, 0.5000],
            [0.0000, 0.1250, 0.2500, 0.3750, 0.5000],
        ], dtype=float)).all()


class TestFilterCutHighlight:
    def test_filter(self, a):
        """Given image data and a threshold, :func:`cut_highlight`
        should set the new whitepoint to the threshold and rebalance
        the contrast.
        """
        threshold = 0.5
        result = f.cut_highlight(a, threshold=threshold)
        assert (np.around(result, 4) == np.array([
            [0.00, 0.50, 1.00, 1.00, 1.00,],
            [0.50, 1.00, 1.00, 1.00, 1.00,],
            [1.00, 1.00, 1.00, 1.00, 1.00,],
            [1.00, 1.00, 1.00, 1.00, 0.50,],
            [1.00, 1.00, 1.00, 0.50, 0.00,],
        ], dtype=float)).all()


class TestFilterCutShadow:
    def test_filter(self, a):
        """Given image data and a threshold, :func:`cut_shadow`
        should make every value in the image data below the
        threshold equal the threshold.
        """
        threshold = 0.5
        result = f.cut_shadow(a, threshold=threshold)
        assert (np.around(result, 4) == np.array([
            [0.00, 0.00, 0.00, 0.50, 1.00,],
            [0.00, 0.00, 0.50, 1.00, 0.50,],
            [0.00, 0.50, 1.00, 0.50, 0.00,],
            [0.50, 1.00, 0.50, 0.00, 0.00,],
            [1.00, 0.50, 0.00, 0.00, 0.00,],
        ], dtype=float)).all()


class TestFilterFlip:
    def test_x_axis(self, a):
        """Given image data and an axis, :func:`flip` flip the
        image around that axis.
        """
        result = f.flip(a, axis=f.X_)
        assert (np.around(result, 4) == np.array([
            [1.0000, 0.7500, 0.5000, 0.2500, 0.0000],
            [0.7500, 1.0000, 0.7500, 0.5000, 0.2500],
            [0.5000, 0.7500, 1.0000, 0.7500, 0.5000],
            [0.2500, 0.5000, 0.7500, 1.0000, 0.7500],
            [0.0000, 0.2500, 0.5000, 0.7500, 1.0000],
        ], dtype=float)).all()

    def test_y_axis(self, a):
        """Given image data and an axis, :func:`flip` flip the
        image around that axis. If the axis is the Y axis, the flip
        happens around the Y axis.
        """
        result = f.flip(a, axis=f.Y)
        assert (np.around(result, 4) == np.array([
            [1.0000, 0.7500, 0.5000, 0.2500, 0.0000],
            [0.7500, 1.0000, 0.7500, 0.5000, 0.2500],
            [0.5000, 0.7500, 1.0000, 0.7500, 0.5000],
            [0.2500, 0.5000, 0.7500, 1.0000, 0.7500],
            [0.0000, 0.2500, 0.5000, 0.7500, 1.0000],
        ], dtype=float)).all()

    def test_z_axis(self, video_2_5_5):
        """Given image data and an axis, :func:`flip` flip the
        image around that axis. If the axis is the Z axis, the flip
        happens around the Z axis.
        """
        result = f.flip(video_2_5_5, axis=f.Z)
        assert (np.around(result, 4) == np.array([
            [
                [1.0000, 0.7500, 0.5000, 0.2500, 0.0000],
                [0.7500, 1.0000, 0.7500, 0.5000, 0.2500],
                [0.5000, 0.7500, 1.0000, 0.7500, 0.5000],
                [0.2500, 0.5000, 0.7500, 1.0000, 0.7500],
                [0.0000, 0.2500, 0.5000, 0.7500, 1.0000],
            ],
            [
                [0.0000, 0.2500, 0.5000, 0.7500, 1.0000],
                [0.2500, 0.5000, 0.7500, 1.0000, 0.7500],
                [0.5000, 0.7500, 1.0000, 0.7500, 0.5000],
                [0.7500, 1.0000, 0.7500, 0.5000, 0.2500],
                [1.0000, 0.7500, 0.5000, 0.2500, 0.0000],
            ],
        ], dtype=float)).all()


class TestFilterGrow:
    def test_filter(self, video_2_3_3):
        """Given image data and a size factor, :func:`glow`
        should zoom into the image by the size factor.
        """
        result = f.grow(video_2_3_3, factor=2)
        assert (np.around(result, 4) == np.array([
            [
                [1.0000, 0.7500, 0.5000, 0.2500, 0.0000, 0.0000],
                [0.7500, 0.5000, 0.2500, 0.2500, 0.2500, 0.2500],
                [0.5000, 0.2500, 0.0000, 0.2500, 0.5000, 0.5000],
                [0.2500, 0.2500, 0.2500, 0.5000, 0.7500, 0.7500],
                [0.0000, 0.2500, 0.5000, 0.7500, 1.0000, 1.0000],
                [0.0000, 0.2500, 0.5000, 0.7500, 1.0000, 1.0000],
            ],
            [
                [1.0000, 0.7500, 0.5000, 0.2500, 0.0000, 0.0000],
                [0.7500, 0.5000, 0.2500, 0.2500, 0.2500, 0.2500],
                [0.5000, 0.2500, 0.0000, 0.2500, 0.5000, 0.5000],
                [0.2500, 0.2500, 0.2500, 0.5000, 0.7500, 0.7500],
                [0.0000, 0.2500, 0.5000, 0.7500, 1.0000, 1.0000],
                [0.0000, 0.2500, 0.5000, 0.7500, 1.0000, 1.0000],
            ],
            [
                [1.0000, 0.7500, 0.5000, 0.2500, 0.0000, 0.0000],
                [0.7500, 0.5000, 0.2500, 0.2500, 0.2500, 0.2500],
                [0.5000, 0.2500, 0.0000, 0.2500, 0.5000, 0.5000],
                [0.2500, 0.2500, 0.2500, 0.5000, 0.7500, 0.7500],
                [0.0000, 0.2500, 0.5000, 0.7500, 1.0000, 1.0000],
                [0.0000, 0.2500, 0.5000, 0.7500, 1.0000, 1.0000],
            ],
            [
                [1.0000, 0.7500, 0.5000, 0.2500, 0.0000, 0.0000],
                [0.7500, 0.5000, 0.2500, 0.2500, 0.2500, 0.2500],
                [0.5000, 0.2500, 0.0000, 0.2500, 0.5000, 0.5000],
                [0.2500, 0.2500, 0.2500, 0.5000, 0.7500, 0.7500],
                [0.0000, 0.2500, 0.5000, 0.7500, 1.0000, 1.0000],
                [0.0000, 0.2500, 0.5000, 0.7500, 1.0000, 1.0000],
            ],
        ], dtype=float)).all()

    def test_image(self, image_1_3_3):
        """Given image data and a size factor, zoom into the image
        by the size factor. This should work on still image data.
        """
        result = f.grow(image_1_3_3, factor=2)
        assert (np.around(result, 4) == np.array([
            [1.0000, 0.7500, 0.5000, 0.2500, 0.0000, 0.0000],
            [0.7500, 0.5000, 0.2500, 0.2500, 0.2500, 0.2500],
            [0.5000, 0.2500, 0.0000, 0.2500, 0.5000, 0.5000],
            [0.2500, 0.2500, 0.2500, 0.5000, 0.7500, 0.7500],
            [0.0000, 0.2500, 0.5000, 0.7500, 1.0000, 1.0000],
            [0.0000, 0.2500, 0.5000, 0.7500, 1.0000, 1.0000],
        ], dtype=float)).all()


class TestFilterInverse:
    def test_filter(self, a):
        """Given image data, :func:`inverse` should invert the
        colors of the image data.
        """
        result = f.inverse(a)
        assert (np.around(result, 4) == np.array([
            [1.0000, 0.7500, 0.5000, 0.2500, 0.0000],
            [0.7500, 0.5000, 0.2500, 0.0000, 0.2500],
            [0.5000, 0.2500, 0.0000, 0.2500, 0.5000],
            [0.2500, 0.0000, 0.2500, 0.5000, 0.7500],
            [0.0000, 0.2500, 0.5000, 0.7500, 1.0000],
        ], dtype=float)).all()


class TestFilterLinearToPolar:
    def test_filter(self, a):
        """Given image data, :func:`linear_to_polar` convert the
        linear coordinates to polar coordinates.
        """
        result = f.linear_to_polar(a)
        assert (np.around(result, 4) == np.array([
            [1.0000, 0.2500, 0.0000, 0.0000, 0.0000],
            [0.2500, 0.5000, 0.7500, 0.5000, 0.2500],
            [0.2500, 0.7500, 1.0000, 0.7500, 0.5000],
            [0.5000, 1.0000, 0.7500, 0.5000, 0.5000],
            [0.5000, 0.7500, 1.0000, 0.7500, 1.0000],
        ], dtype=float)).all()

    def test_video(self, video_2_5_5):
        """Given image data, :func:`linear_to_polar` convert the
        linear coordinates to polar coordinates. This should also work
        for video.
        """
        result = f.linear_to_polar(video_2_5_5)
        assert (np.around(result, 4) == np.array([
            [
                [0.0000, 0.2500, 0.0000, 0.0000, 0.0000],
                [0.2500, 0.5000, 0.7500, 0.5000, 0.2500],
                [0.2500, 0.7500, 1.0000, 0.7500, 0.5000],
                [0.5000, 1.0000, 0.7500, 0.5000, 0.5000],
                [0.5000, 0.7500, 1.0000, 0.7500, 1.0000],
            ],
            [
                [1.0000, 0.7500, 1.0000, 1.0000, 1.0000],
                [0.7500, 1.0000, 0.7500, 0.5000, 0.7500],
                [0.7500, 0.7500, 0.5000, 0.2500, 0.5000],
                [0.5000, 1.0000, 0.7500, 1.0000, 0.5000],
                [0.5000, 0.7500, 1.0000, 0.7500, 0.5000],
            ],
        ], dtype=float)).all()


class TestFilterPinch:
    def test_filter(self, a):
        """Given image data, an amount of the pinch, a radius, a
        scale, and an offset, :func:`pinch` should perform
        a pinch on the image data.
        """
        result = f.pinch(
            a,
            amount=0.5,
            radius=3.0,
            scale=(0.5, 0.5),
            offset=(0, 0, 0)
        )
        assert (np.around(result, 4) == np.array([
            [0.0000, 0.0859, 0.1465, 0.2441, 0.3438],
            [0.0859, 0.2188, 0.4609, 0.8340, 0.3896],
            [0.1465, 0.4609, 0.6719, 0.7500, 0.0713],
            [0.2441, 0.8340, 0.7500, 0.1719, 0.0225],
            [0.3438, 0.3896, 0.0713, 0.0225, 0.0000],
        ], dtype=float)).all()

    def test_video(self, video_2_5_5):
        """Given image data, :func:`linear_to_polar` convert the
        linear coordinates to polar coordinates. This should also work
        for video.
        """
        result = f.pinch(
            video_2_5_5,
            amount=0.5,
            radius=3.0,
            scale=(0.5, 0.5),
            offset=(0, 0, 0)
        )
        assert (np.around(result, 4) == np.array([
            [
                [0.0000, 0.0859, 0.1465, 0.2441, 0.3438],
                [0.0859, 0.2188, 0.4609, 0.8340, 0.3896],
                [0.1465, 0.4609, 0.6719, 0.7500, 0.0713],
                [0.2441, 0.8340, 0.7500, 0.1719, 0.0225],
                [0.3438, 0.3896, 0.0713, 0.0225, 0.0000],
            ],
            [
                [0.5166, 0.4141, 0.1660, 0.0684, 0.0000],
                [0.4141, 0.8770, 0.6016, 0.2109, 0.0479],
                [0.1660, 0.6016, 0.8872, 0.4219, 0.0537],
                [0.0684, 0.2109, 0.4219, 0.8872, 0.1025],
                [0.0000, 0.0479, 0.0537, 0.1025, 0.1914],
            ],
        ], dtype=float)).all()


class TestFilterPolarToLinear:
    def test_filter(self, a):
        """Given image data, :func:`polar_to_linear` should convert
        the polar coordinates to linear coordinates.
        """
        result = f.polar_to_linear(a)
        assert (np.around(result, 4) == np.array([
            [1.0000, 0.7500, 0.5000, 0.0000, 0.0000],
            [1.0000, 0.5000, 0.2500, 0.0000, 0.0000],
            [1.0000, 0.7500, 1.0000, 0.7500, 1.0000],
            [1.0000, 1.0000, 0.7500, 0.5000, 0.2500],
            [1.0000, 0.7500, 1.0000, 0.7500, 0.7500],
        ], dtype=float)).all()

    def test_video(self, video_2_5_5):
        """Given image data, :func:`linear_to_polar` convert the
        linear coordinates to polar coordinates. This should also work
        for video.
        """
        result = f.polar_to_linear(video_2_5_5)
        assert (np.around(result, 4) == np.array([
            [
                [1.0000, 0.7500, 0.5000, 0.0000, 0.0000],
                [1.0000, 0.5000, 0.2500, 0.0000, 0.0000],
                [1.0000, 0.7500, 1.0000, 0.7500, 1.0000],
                [1.0000, 1.0000, 0.7500, 0.5000, 0.2500],
                [1.0000, 0.7500, 1.0000, 0.7500, 0.7500],
            ],
            [
                [1.0000, 0.7500, 0.5000, 0.0000, 0.0000],
                [1.0000, 1.0000, 0.7500, 0.0000, 0.0000],
                [1.0000, 0.7500, 0.5000, 0.2500, 0.0000],
                [1.0000, 1.0000, 0.7500, 1.0000, 0.7500],
                [1.0000, 0.7500, 0.5000, 0.2500, 0.2500],
            ],
        ], dtype=float)).all()


class TestFilterPosterize:
    def test_filter(self, a):
        """Given image data, :func:`posterize` should reduce the
        number of colors in the image data.
        """
        result = f.posterize(a, levels=3)
        assert (np.around(result, 4) == np.array([
            [0.0000, 0.0000, 0.5000, 1.0000, 1.0000],
            [0.0000, 0.5000, 1.0000, 1.0000, 1.0000],
            [0.5000, 1.0000, 1.0000, 1.0000, 0.5000],
            [1.0000, 1.0000, 1.0000, 0.5000, 0.0000],
            [1.0000, 1.0000, 0.5000, 0.0000, 0.0000],
        ], dtype=float)).all()

    def test_video(self, video_2_5_5):
        """Given image data, :func:`posterize` should reduce the
        number of colors in the image data. This should also work
        for video.
        """
        result = f.posterize(video_2_5_5, levels=3)
        assert (np.around(result, 4) == np.array([
            [
                [0.0000, 0.0000, 0.5000, 1.0000, 1.0000],
                [0.0000, 0.5000, 1.0000, 1.0000, 1.0000],
                [0.5000, 1.0000, 1.0000, 1.0000, 0.5000],
                [1.0000, 1.0000, 1.0000, 0.5000, 0.0000],
                [1.0000, 1.0000, 0.5000, 0.0000, 0.0000],
            ],
            [
                [1.0000, 1.0000, 0.5000, 0.0000, 0.0000],
                [1.0000, 1.0000, 1.0000, 0.5000, 0.0000],
                [0.5000, 1.0000, 1.0000, 1.0000, 0.5000],
                [0.0000, 0.5000, 1.0000, 1.0000, 1.0000],
                [0.0000, 0.0000, 0.5000, 1.0000, 1.0000],
            ],
        ], dtype=float)).all()


class TestFilterRipple:
    def test_filter(self, a):
        """Given image data, :func:`ripple` should convert
        the polar coordinates to linear coordinates.
        """
        result = f.ripple(
            a,
            wave=(2, 2),
            amp=(2, 2),
            distaxis=(f.Y_, f.X_),
            offset=(0, 0)
        )
        assert (np.around(result, 4) == np.array([
            [1.0000, 0.0000, 0.5000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.7500, 0.0000, 0.7500],
            [0.5000, 0.7500, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.5000, 0.0000],
            [0.0000, 0.7500, 0.0000, 0.0000, 0.0000],
        ], dtype=float)).all()

    def test_video(self, video_2_5_5):
        """Given image data, :func:`ripple` convert the
        linear coordinates to polar coordinates. This should also work
        for video.
        """
        result = f.ripple(
            video_2_5_5,
            wave=(2, 2),
            amp=(2, 2),
            distaxis=(f.Y_, f.X_),
            offset=(0, 0)
        )
        assert (np.around(result, 4) == np.array([
            [
                [1.0000, 0.0000, 0.5000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.7500, 0.0000, 0.7500],
                [0.5000, 0.7500, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.5000, 0.0000],
                [0.0000, 0.7500, 0.0000, 0.0000, 0.0000],
            ],
            [
                [1.0000, 0.0000, 0.5000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.2500, 0.0000, 0.7500],
                [0.5000, 0.2500, 1.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.0000, 1.0000, 0.0000],
                [0.0000, 0.7500, 0.0000, 0.0000, 0.0000],
            ],
        ], dtype=float)).all()


class TestFilterRotate2d:
    def test_filter(self, a):
        """Given image data and an angle, :func:`rotate_2d`
        should rotate the image data by that amount in the
        clockwise direction.
        """
        result = f.rotate_2d(a, 45.0)
        assert (np.around(result, 4) == np.array([
            [
                [0.0938, 0.5947, 0.8794, 0.5947, 0.0781],
                [0.2803, 0.6484, 0.8989, 0.6484, 0.2803],
                [0.2969, 0.6406, 1.0000, 0.6406, 0.2969],
                [0.2803, 0.6484, 0.8989, 0.6484, 0.2803],
                [0.0938, 0.5947, 0.8794, 0.5947, 0.0781],
            ],
        ], dtype=float)).all()

    def test_filter_origin(self, a):
        """Given image data and an angle, :func:`rotate_2d`
        should rotate the image data by that amount in the
        clockwise direction. If an origin is given, the
        image should be rotated around that point.
        """
        result = f.rotate_2d(a, 45.0, origin=(1, 1))
        assert (np.around(result, 4) == np.array([
            [
                [0.1484, 0.5000, 0.8516, 0.7891, 0.4375],
                [0.1406, 0.5000, 0.8594, 0.7969, 0.4375],
                [0.1484, 0.5000, 0.8516, 0.7891, 0.4375],
                [0.0000, 0.3572, 0.8340, 0.7891, 0.2673],
                [0.0000, 0.0000, 0.5706, 0.4358, 0.0000],
            ],
        ], dtype=float)).all()

    def test_filter_video(self, video_2_5_5):
        """Given image data and an angle, :func:`rotate_2d`
        should rotate the image data by that amount in the
        clockwise direction. If given video data, each frame
        of the data should be rotated.
        """
        result = f.rotate_2d(video_2_5_5, 45.0)
        assert (np.around(result, 4) == np.array([
            [
                [
                    [0.0938, 0.5947, 0.8794, 0.5947, 0.0781],
                    [0.2803, 0.6484, 0.8989, 0.6484, 0.2803],
                    [0.2969, 0.6406, 1.0000, 0.6406, 0.2969],
                    [0.2803, 0.6484, 0.8989, 0.6484, 0.2803],
                    [0.0938, 0.5947, 0.8794, 0.5947, 0.0781],
                ],
                [
                    [0.0938, 0.2803, 0.2969, 0.2803, 0.0781],
                    [0.5947, 0.6484, 0.6406, 0.6484, 0.5947],
                    [0.8794, 0.8989, 1.0000, 0.8989, 0.8794],
                    [0.5947, 0.6484, 0.6406, 0.6484, 0.5947],
                    [0.0938, 0.2803, 0.2969, 0.2803, 0.0781],
                ],
            ],
        ], dtype=float)).all()


class TestFilterRotate90:
    def test_filter(self, image_5_5_tenths):
        """Given image data and a direction, :func:`rotate_90`
        should rotate the image data 90° in that direction.
        """
        result = f.rotate_90(image_5_5_tenths)
        assert (np.around(result, 4) == np.array([
            [
                [0.8000, 0.6000, 0.4000, 0.2000, 0.0000],
                [0.9000, 0.7000, 0.5000, 0.3000, 0.1000],
                [1.0000, 0.8000, 0.6000, 0.4000, 0.2000],
                [0.7000, 0.9000, 0.7000, 0.5000, 0.3000],
                [0.8000, 1.0000, 0.8000, 0.6000, 0.4000],
            ],
        ], dtype=float)).all()

    def test_ccw(self, image_5_5_tenths):
        """Given image data and a direction, :func:`rotate_90`
        rotate the image data 90° in that direction.
        """
        result = f.rotate_90(image_5_5_tenths, direction='ccw')
        assert (np.around(result, 4) == np.array([
            [
                [0.4000, 0.6000, 0.8000, 1.0000, 0.8000],
                [0.3000, 0.5000, 0.7000, 0.9000, 0.7000],
                [0.2000, 0.4000, 0.6000, 0.8000, 1.0000],
                [0.1000, 0.3000, 0.5000, 0.7000, 0.9000],
                [0.0000, 0.2000, 0.4000, 0.6000, 0.8000],
            ],
        ], dtype=float)).all()


class TestFilterSkew:
    def test_filter(self, a):
        """Given image data and a slope, :func:`skew` should
        skew the image data by an amount equal to the slope.
        """
        result = f.skew(a, slope=2.0)
        assert (np.around(result, 4) == np.array([
            [0.0000, 0.2500, 0.5000, 0.7500, 1.0000],
            [1.0000, 0.7500, 0.2500, 0.5000, 0.7500],
            [0.7500, 1.0000, 0.7500, 0.5000, 0.5000],
            [0.2500, 0.7500, 1.0000, 0.7500, 0.5000],
            [0.5000, 0.2500, 0.0000, 1.0000, 0.7500],
        ], dtype=float)).all()

    def test_video(self, video_2_5_5):
        """Given image data and a slope, :func:`skew` should
        skew the image data by an amount equal to the slope. This
        should also work for video.
        """
        result = f.skew(video_2_5_5, slope=2.0)
        assert (np.around(result, 4) == np.array([
            [
                [0.0000, 0.2500, 0.5000, 0.7500, 1.0000],
                [1.0000, 0.7500, 0.2500, 0.5000, 0.7500],
                [0.7500, 1.0000, 0.7500, 0.5000, 0.5000],
                [0.2500, 0.7500, 1.0000, 0.7500, 0.5000],
                [0.5000, 0.2500, 0.0000, 1.0000, 0.7500],
            ],
            [
                [1.0000, 0.7500, 0.5000, 0.2500, 0.0000],
                [0.5000, 0.2500, 0.7500, 1.0000, 0.7500],
                [0.7500, 1.0000, 0.7500, 0.5000, 0.5000],
                [0.7500, 0.2500, 0.5000, 0.7500, 1.0000],
                [0.5000, 0.7500, 1.0000, 0.0000, 0.2500],
            ],
        ], dtype=float)).all()


class TestFilterSkew:
    def test_filter(self, a):
        """Given image data, a radius, a strength, and an offset,
        :func:`skew` should perform a twirl distortion on
        the data.
        """
        result = f.twirl(a, radius=5.0, strength=0.25)
        assert (np.around(result, 4) == np.array([
            [0.0019, 0.2537, 0.5047, 0.7547, 0.9963],
            [0.2491, 0.5001, 0.7565, 0.9871, 0.7499],
            [0.4969, 0.7438, 0.9785, 0.7275, 0.4935],
            [0.7468, 0.9873, 0.7715, 0.5010, 0.2438],
            [0.9963, 0.7586, 0.5129, 0.2627, 0.0088],
        ], dtype=float)).all()

    def test_video(self, video_2_5_5):
        """Given image data, a radius, a strength, and an offset,
        :func:`skew` should perform a twirl distortion on
        the data. This should also work for video.
        """
        result = f.twirl(video_2_5_5, radius=5.0, strength=0.25)
        assert (np.around(result, 4) == np.array([
            [
                [0.0019, 0.2537, 0.5047, 0.7547, 0.9963],
                [0.2491, 0.5001, 0.7565, 0.9871, 0.7499],
                [0.4969, 0.7438, 0.9785, 0.7275, 0.4935],
                [0.7468, 0.9873, 0.7715, 0.5010, 0.2438],
                [0.9963, 0.7586, 0.5129, 0.2627, 0.0088],
            ],
            [
                [0.9981, 0.7491, 0.4968, 0.2469, 0.0037],
                [0.7537, 0.9912, 0.7373, 0.4938, 0.2588],
                [0.5047, 0.7626, 0.9775, 0.7510, 0.5127],
                [0.2547, 0.5065, 0.7510, 0.9775, 0.7626],
                [0.0037, 0.2501, 0.4938, 0.7435, 0.9914],
            ],
        ], dtype=float)).all()

    def test_video_offset(self, video_2_5_5):
        """Given image data, a radius, a strength, and an offset,
        :func:`skew` should perform a twirl distortion on
        the data. This should also work for video. The offset
        should move the center of the effect by the amount given.
        """
        result = f.twirl(
            video_2_5_5, radius=5.0, strength=0.25, offset=(-2, 2)
        )
        assert (np.around(result, 4) == np.array([
            [
                [0.0005, 0.2515, 0.5047, 0.7626, 0.9785],
                [0.2496, 0.4985, 0.7453, 0.9873, 0.7715],
                [0.4998, 0.7487, 0.9963, 0.7586, 0.5129],
                [0.7499, 0.9992, 0.7519, 0.5037, 0.2547],
                [0.9999, 0.7503, 0.5008, 0.2513, 0.0015],
            ],
            [
                [0.9995, 0.7511, 0.5031, 0.2562, 0.0225],
                [0.7505, 0.9985, 0.7468, 0.4935, 0.2490],
                [0.5004, 0.7505, 0.9963, 0.7499, 0.5062],
                [0.2503, 0.5001, 0.7500, 0.9963, 0.7531],
                [0.0001, 0.2500, 0.4999, 0.7495, 0.9985],
            ],
        ], dtype=float)).all()
