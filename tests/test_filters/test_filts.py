"""
test_filts
~~~~~~~~~~

Unit tests for :mod:`pjimg.filters.value`.
"""
import numpy as np
import pytest as pt

from pjimg.filters import value as f
from tests.fixtures import *


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


class TestDistance:
    def test_filter(self, a):
        """Given image data, :func:`distance`
        should make every value in the image data its
        relative distance to the nearest black value.
        """
        result = f.distance(a)
        assert (np.around(result, 4) == np.array([
            [0.0000, 0.2500, 0.5000, 0.7500, 1.0000],
            [0.2500, 0.3500, 0.5492, 0.7992, 0.7500],
            [0.5000, 0.5492, 0.7000, 0.5492, 0.5000],
            [0.7500, 0.7992, 0.5492, 0.3500, 0.2500],
            [1.0000, 0.7500, 0.5000, 0.2500, 0.0000],
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
