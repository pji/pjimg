"""
test_distort
~~~~~~~~~~~~

Unit tests for :mod:`pjimg.filters.distort`.
"""
import numpy as np
import pytest as pt

from pjimg.filters import distort as f
from tests.fixtures import a, video_2_5_5


# Test Cases.
class TestLinearToPolar:
    def test_filter(self, a):
        """Given image data, :func:`linear_to_polar` convert the
        linear coordinates to polar coordinates.
        """
        result = f.linear_to_polar(a)
        assert (np.around(result, 4) == np.array([
            [0.0000, 0.2500, 0.0000, 0.0000, 0.0000],
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
                [0.0000, 0.7500, 1.0000, 1.0000, 1.0000],
                [0.7500, 1.0000, 0.7500, 0.5000, 0.7500],
                [0.7500, 0.7500, 0.5000, 0.2500, 0.5000],
                [0.5000, 1.0000, 0.7500, 1.0000, 0.5000],
                [0.5000, 0.7500, 1.0000, 0.7500, 0.5000],
            ],
        ], dtype=float)).all()


class TestPinch:
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


class TestPolarToLinear:
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


class TestRipple:
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


class TestTwirl:
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
