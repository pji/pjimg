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
class TestFilterBoxBlur:
    def test_filter(self, a):
        """Given image data and a box size, :func:`filter_box_blur`
         perform a box blur on the image data.
        """
        result = f.filter_box_blur(a, size=2)
        assert (np.around(result, 4) == np.array([
            [0.2500, 0.2500, 0.5000, 0.7500, 0.8750],
            [0.2500, 0.2500, 0.5000, 0.7500, 0.8750],
            [0.5000, 0.5000, 0.7500, 0.8750, 0.7500],
            [0.7500, 0.7500, 0.8750, 0.7500, 0.5000],
            [0.8750, 0.8750, 0.7500, 0.5000, 0.2500],
        ], dtype=float)).all()

    def test_filter_video(self, video_2_5_5):
        """Given three dimensional image data, :func:`filter_box_blur`
        the blur should be performed on all frames of the image data.
        """
        result = f.filter_box_blur(video_2_5_5, size=2)
        assert (np.around(result, 4) == np.array([
            [
                [0.2500, 0.2500, 0.5000, 0.7500, 0.8750],
                [0.2500, 0.2500, 0.5000, 0.7500, 0.8750],
                [0.5000, 0.5000, 0.7500, 0.8750, 0.7500],
                [0.7500, 0.7500, 0.8750, 0.7500, 0.5000],
                [0.8750, 0.8750, 0.7500, 0.5000, 0.2500],
            ],
            [
                [0.8750, 0.8750, 0.7500, 0.5000, 0.2500],
                [0.8750, 0.8750, 0.7500, 0.5000, 0.2500],
                [0.7500, 0.7500, 0.8750, 0.7500, 0.5000],
                [0.5000, 0.5000, 0.7500, 0.8750, 0.7500],
                [0.2500, 0.2500, 0.5000, 0.7500, 0.8750],
            ],
        ], dtype=float)).all()


class TestFilterColorize:
    def test_filter(self, image_1_3_3):
        """Given an RGB color and grayscale image data,
        :func:`filter_colorize` should apply the color to
        the image data.
        """
        result = f.filter_colorize(
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

    def test_filter_by_colorkey(self, image_1_3_3):
        """Given an color key and grayscale image data,
        :func:`filter_colorize` should apply the color to
        the image data.
        """
        result = f.filter_colorize(
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

    def test_filter_on_video(self, video_2_3_3):
        """Given an RGB color and grayscale image data,
        :func:`filter_colorize` should apply the color to
        the video data.
        """
        result = f.filter_colorize(
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
        """Given image data, :func:`filter_contrast` adjust the
        range of the data to ensure the darkest color is black
        and the lightest is white.
        """
        result = f.filter_contrast(image_5_5_low_contrast)
        assert (np.around(result, 4) == np.array([
            [0.0000, 0.2500, 0.5000, 0.7500, 1.0000],
            [0.0000, 0.2500, 0.5000, 0.7500, 1.0000],
            [0.0000, 0.2500, 0.5000, 0.7500, 1.0000],
            [0.0000, 0.2500, 0.5000, 0.7500, 1.0000],
            [0.0000, 0.2500, 0.5000, 0.7500, 1.0000],
        ], dtype=float)).all()

    def test_filter_black(self, image_5_5_low_contrast):
        """Given image data and a black point, :func:`filter_contrast`
        adjust the range of the data to ensure the darkest color is the
        given black point and the lightest is white.
        """
        result = f.filter_contrast(image_5_5_low_contrast, black=0.5)
        assert (np.around(result, 4) == np.array([
            [0.5000, 0.6250, 0.7500, 0.8750, 1.0000],
            [0.5000, 0.6250, 0.7500, 0.8750, 1.0000],
            [0.5000, 0.6250, 0.7500, 0.8750, 1.0000],
            [0.5000, 0.6250, 0.7500, 0.8750, 1.0000],
            [0.5000, 0.6250, 0.7500, 0.8750, 1.0000],
        ], dtype=float)).all()

    def test_filter_white(self, image_5_5_low_contrast):
        """Given image data and a white point, :func:`filter_contrast`
        adjust the range of the data to ensure the darkest color is black
        and the lightest is the given maximum.
        """
        result = f.filter_contrast(image_5_5_low_contrast, white=0.5)
        assert (np.around(result, 4) == np.array([
            [0.0000, 0.1250, 0.2500, 0.3750, 0.5000],
            [0.0000, 0.1250, 0.2500, 0.3750, 0.5000],
            [0.0000, 0.1250, 0.2500, 0.3750, 0.5000],
            [0.0000, 0.1250, 0.2500, 0.3750, 0.5000],
            [0.0000, 0.1250, 0.2500, 0.3750, 0.5000],
        ], dtype=float)).all()


class TestFilterFlip:
    def test_filter_x_axis(self, a):
        """Given image data and an axis, :func:`filter_flip` flip the
        image around that axis.
        """
        result = f.filter_flip(a, axis=f.X_)
        assert (np.around(result, 4) == np.array([
            [1.0000, 0.7500, 0.5000, 0.2500, 0.0000],
            [0.7500, 1.0000, 0.7500, 0.5000, 0.2500],
            [0.5000, 0.7500, 1.0000, 0.7500, 0.5000],
            [0.2500, 0.5000, 0.7500, 1.0000, 0.7500],
            [0.0000, 0.2500, 0.5000, 0.7500, 1.0000],
        ], dtype=float)).all()

    def test_filter_y_axis(self, a):
        """Given image data and an axis, :func:`filter_flip` flip the
        image around that axis. If the axis is the Y axis, the flip
        happens around the Y axis.
        """
        result = f.filter_flip(a, axis=f.Y)
        assert (np.around(result, 4) == np.array([
            [1.0000, 0.7500, 0.5000, 0.2500, 0.0000],
            [0.7500, 1.0000, 0.7500, 0.5000, 0.2500],
            [0.5000, 0.7500, 1.0000, 0.7500, 0.5000],
            [0.2500, 0.5000, 0.7500, 1.0000, 0.7500],
            [0.0000, 0.2500, 0.5000, 0.7500, 1.0000],
        ], dtype=float)).all()

    def test_filter_z_axis(self, video_2_5_5):
        """Given image data and an axis, :func:`filter_flip` flip the
        image around that axis. If the axis is the Z axis, the flip
        happens around the Z axis.
        """
        result = f.filter_flip(video_2_5_5, axis=f.Z)
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


class TestFilterGaussianBlue:
    def test_filter(self, a):
        """Given image data and a sigma, :func:`filter_gaussian_blur`
        should perform a gaussian blur on the image data.
        """
        result = f.filter_gaussian_blur(a, sigma=0.5)
        assert (np.around(result, 4) == np.array([
            [0.1070, 0.3036, 0.5534, 0.7918, 0.9158],
            [0.3036, 0.5002, 0.7442, 0.9046, 0.7918],
            [0.5534, 0.7442, 0.9044, 0.7442, 0.5534],
            [0.7918, 0.9046, 0.7442, 0.5002, 0.3036],
            [0.9158, 0.7918, 0.5534, 0.3036, 0.1070],
        ], dtype=float)).all()

    def test_filter_video(self, video_2_5_5):
        """Given image data and a sigma, :func:`filter_gaussian_blur`
        should perform a gaussian blur on the image data.
        """
        result = f.filter_gaussian_blur(video_2_5_5, sigma=0.5)
        assert (np.around(result, 4) == np.array([
            [
                [0.1070, 0.3036, 0.5534, 0.7918, 0.9158],
                [0.3036, 0.5002, 0.7442, 0.9046, 0.7918],
                [0.5534, 0.7442, 0.9044, 0.7442, 0.5534],
                [0.7918, 0.9046, 0.7442, 0.5002, 0.3036],
                [0.9158, 0.7918, 0.5534, 0.3036, 0.1070],
            ],
            [
                [0.9158, 0.7918, 0.5534, 0.3036, 0.1070],
                [0.7918, 0.9046, 0.7442, 0.5002, 0.3036],
                [0.5534, 0.7442, 0.9044, 0.7442, 0.5534],
                [0.3036, 0.5002, 0.7442, 0.9046, 0.7918],
                [0.1070, 0.3036, 0.5534, 0.7918, 0.9158],
            ],
        ], dtype=float)).all()


class TestFilterGlow:
    def test_filter(self, video_2_5_5):
        """Given image data and a size factor, :func:`filter_glow`
        should zoom into the image by the size factor.
        """
        result = f.filter_glow(video_2_5_5, sigma=4)
        assert (np.around(result, 4) == np.array([
            [
                [0.7802, 0.8597, 0.9389, 0.9813, 1.0000],
                [0.8597, 0.9211, 0.9736, 1.0000, 0.9813],
                [0.9389, 0.9736, 1.0000, 0.9736, 0.9389],
                [0.9813, 1.0000, 0.9736, 0.9211, 0.8597],
                [1.0000, 0.9813, 0.9389, 0.8597, 0.7802],
            ],
            [
                [1.0000, 0.9813, 0.9389, 0.8597, 0.7802],
                [0.9813, 1.0000, 0.9736, 0.9211, 0.8597],
                [0.9389, 0.9736, 1.0000, 0.9736, 0.9389],
                [0.8597, 0.9211, 0.9736, 1.0000, 0.9813],
                [0.7802, 0.8597, 0.9389, 0.9813, 1.0000],
            ],
        ], dtype=float)).all()


class TestFilterGrow:
    def test_filter(self, video_2_3_3):
        """Given image data and a size factor, :func:`filter_glow`
        should zoom into the image by the size factor.
        """
        result = f.filter_grow(video_2_3_3, factor=2)
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

    def test_filter_image(self, image_1_3_3):
        """Given image data and a size factor, zoom into the image
        by the size factor. This should work on still image data.
        """
        result = f.filter_grow(image_1_3_3, factor=2)
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
        """Given image data, :func:`filter_inverse` should invert the
        colors of the image data.
        """
        result = f.filter_inverse(a)
        assert (np.around(result, 4) == np.array([
            [1.0000, 0.7500, 0.5000, 0.2500, 0.0000],
            [0.7500, 0.5000, 0.2500, 0.0000, 0.2500],
            [0.5000, 0.2500, 0.0000, 0.2500, 0.5000],
            [0.2500, 0.0000, 0.2500, 0.5000, 0.7500],
            [0.0000, 0.2500, 0.5000, 0.7500, 1.0000],
        ], dtype=float)).all()


class TestFilterLinearToPolar:
    def test_filter(self, a):
        """Given image data, :func:`filter_linear_to_polar` convert the
        linear coordinates to polar coordinates.
        """
        result = f.filter_linear_to_polar(a)
        assert (np.around(result, 4) == np.array([
            [1.0000, 0.2500, 0.0000, 0.0000, 0.0000],
            [0.2500, 0.5000, 0.7500, 0.5000, 0.2500],
            [0.2500, 0.7500, 1.0000, 0.7500, 0.5000],
            [0.5000, 1.0000, 0.7500, 0.5000, 0.5000],
            [0.5000, 0.7500, 1.0000, 0.7500, 1.0000],
        ], dtype=float)).all()

    def test_filter_video(self, video_2_5_5):
        """Given image data, :func:`filter_linear_to_polar` convert the
        linear coordinates to polar coordinates. This should also work
        for video.
        """
        result = f.filter_linear_to_polar(video_2_5_5)
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


class TestFilterMotionBlur:
    def test_filter(self, a):
        """Given image data, an amount, and a direction,
        :func:`motion_blur` should perform a motion blur
        on the image data.
        """
        result = f.filter_motion_blur(a, amount=2, axis=f.X_)
        assert (np.around(result, 4) == np.array([
            [0.1250, 0.1250, 0.3750, 0.6250, 0.8750],
            [0.3750, 0.3750, 0.6250, 0.8750, 0.8750],
            [0.6250, 0.6250, 0.8750, 0.8750, 0.6250],
            [0.8750, 0.8750, 0.8750, 0.6250, 0.3750],
            [0.8750, 0.8750, 0.6250, 0.3750, 0.1250],
        ], dtype=float)).all()

    def test_filter_vertical(self, a):
        """Given image data, an amount, and a direction,
        :func:`motion_blur` should perform a motion blur
        on the image data. If the direction is the Y axis,
        the blur should be vertical.
        """
        result = f.filter_motion_blur(a, amount=2, axis=f.Y_)
        assert (np.around(result, 4) == np.array([
            [0.1250, 0.3750, 0.6250, 0.8750, 0.8750],
            [0.1250, 0.3750, 0.6250, 0.8750, 0.8750],
            [0.3750, 0.6250, 0.8750, 0.8750, 0.6250],
            [0.6250, 0.8750, 0.8750, 0.6250, 0.3750],
            [0.8750, 0.8750, 0.6250, 0.3750, 0.1250],
        ], dtype=float)).all()

    def test_filter_video(self, video_2_5_5):
        """Given image data, an amount, and a direction,
        :func:`motion_blur` should perform a motion blur
        on the video data.
        """
        result = f.filter_motion_blur(video_2_5_5, amount=2, axis=f.X_)
        assert (np.around(result, 4) == np.array([
            [
                [0.1250, 0.1250, 0.3750, 0.6250, 0.8750],
                [0.3750, 0.3750, 0.6250, 0.8750, 0.8750],
                [0.6250, 0.6250, 0.8750, 0.8750, 0.6250],
                [0.8750, 0.8750, 0.8750, 0.6250, 0.3750],
                [0.8750, 0.8750, 0.6250, 0.3750, 0.1250],
            ],
            [
                [0.8750, 0.8750, 0.6250, 0.3750, 0.1250],
                [0.8750, 0.8750, 0.8750, 0.6250, 0.3750],
                [0.6250, 0.6250, 0.8750, 0.8750, 0.6250],
                [0.3750, 0.3750, 0.6250, 0.8750, 0.8750],
                [0.1250, 0.1250, 0.3750, 0.6250, 0.8750],
            ],
        ], dtype=float)).all()

    def test_filter_invalid_axis(self, a):
        """If given an invalid axis, :func:`filter_motion_blur` should
        raise a :class:`ValueError` exception.
        """
        with pt.raises(
            ValueError, match='motion_blur can only affect the X or Y axis.'
        ):
            _ = f.filter_motion_blur(a, amount=2, axis=f.Z)


class TestFilterPinch:
    def test_filter(self, a):
        """Given image data, an amount of the pinch, a radius, a
        scale, and an offset, :func:`filter_pinch` should perform
        a pinch on the image data.
        """
        result = f.filter_pinch(
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

    def test_filter_video(self, video_2_5_5):
        """Given image data, :func:`filter_linear_to_polar` convert the
        linear coordinates to polar coordinates. This should also work
        for video.
        """
        result = f.filter_pinch(
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
        result = f.filter_polar_to_linear(a)
        assert (np.around(result, 4) == np.array([
            [1.0000, 0.7500, 0.5000, 0.0000, 0.0000],
            [1.0000, 0.5000, 0.2500, 0.0000, 0.0000],
            [1.0000, 0.7500, 1.0000, 0.7500, 1.0000],
            [1.0000, 1.0000, 0.7500, 0.5000, 0.2500],
            [1.0000, 0.7500, 1.0000, 0.7500, 0.7500],
        ], dtype=float)).all()

    def test_filter_video(self, video_2_5_5):
        """Given image data, :func:`filter_linear_to_polar` convert the
        linear coordinates to polar coordinates. This should also work
        for video.
        """
        result = f.filter_polar_to_linear(video_2_5_5)
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


class TestFilterRipple:
    def test_filter(self, a):
        """Given image data, :func:`filter_ripple` should convert
        the polar coordinates to linear coordinates.
        """
        result = f.filter_ripple(
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

    def test_filter_video(self, video_2_5_5):
        """Given image data, :func:`filter_ripple` convert the
        linear coordinates to polar coordinates. This should also work
        for video.
        """
        result = f.filter_ripple(
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


class TestFilterRotate90:
    def test_filter(self, image_5_5_tenths):
        """Given image data and a direction, :func:`filter_rotate_90`
        should rotate the image data 90° in that direction.
        """
        result = f.filter_rotate_90(image_5_5_tenths)
        assert (np.around(result, 4) == np.array([
            [
                [0.8000, 0.6000, 0.4000, 0.2000, 0.0000],
                [0.9000, 0.7000, 0.5000, 0.3000, 0.1000],
                [1.0000, 0.8000, 0.6000, 0.4000, 0.2000],
                [0.7000, 0.9000, 0.7000, 0.5000, 0.3000],
                [0.8000, 1.0000, 0.8000, 0.6000, 0.4000],
            ],
        ], dtype=float)).all()

    def test_filter_ccw(self, image_5_5_tenths):
        """Given image data and a direction, :func:`filter_rotate_90`
        rotate the image data 90° in that direction.
        """
        result = f.filter_rotate_90(image_5_5_tenths, direction='ccw')
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
        """Given image data and a slope, :func:`filter_skew` should
        skew the image data by an amount equal to the slope.
        """
        result = f.filter_skew(a, slope=2.0)
        assert (np.around(result, 4) == np.array([
            [0.0000, 0.2500, 0.5000, 0.7500, 1.0000],
            [1.0000, 0.7500, 0.2500, 0.5000, 0.7500],
            [0.7500, 1.0000, 0.7500, 0.5000, 0.5000],
            [0.2500, 0.7500, 1.0000, 0.7500, 0.5000],
            [0.5000, 0.2500, 0.0000, 1.0000, 0.7500],
        ], dtype=float)).all()

    def test_filter_video(self, video_2_5_5):
        """Given image data and a slope, :func:`filter_skew` should
        skew the image data by an amount equal to the slope. This
        should also work for video.
        """
        result = f.filter_skew(video_2_5_5, slope=2.0)
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
        :func:`filter_skew` should perform a twirl distortion on
        the data.
        """
        result = f.filter_twirl(a, radius=5.0, strength=0.25)
        assert (np.around(result, 4) == np.array([
            [0.0019, 0.2537, 0.5047, 0.7547, 0.9963],
            [0.2491, 0.5001, 0.7565, 0.9871, 0.7499],
            [0.4969, 0.7438, 0.9785, 0.7275, 0.4935],
            [0.7468, 0.9873, 0.7715, 0.5010, 0.2438],
            [0.9963, 0.7586, 0.5129, 0.2627, 0.0088],
        ], dtype=float)).all()

    def test_filter_video(self, video_2_5_5):
        """Given image data, a radius, a strength, and an offset,
        :func:`filter_skew` should perform a twirl distortion on
        the data. This should also work for video.
        """
        result = f.filter_twirl(video_2_5_5, radius=5.0, strength=0.25)
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

    def test_filter_video_offset(self, video_2_5_5):
        """Given image data, a radius, a strength, and an offset,
        :func:`filter_skew` should perform a twirl distortion on
        the data. This should also work for video. The offset
        should move the center of the effect by the amount given.
        """
        result = f.filter_twirl(
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
