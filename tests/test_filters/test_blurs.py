"""
test_blurs
~~~~~~~~~~

Unit tests for pjimg.sources.blurs
"""
import numpy as np
import pytest as pt

import pjimg.filters.blurs as f
from tests.fixtures import a, video_2_5_5


# Test cases.
class TestBoxBlur:
    def test_filter(self, a):
        """Given image data and a box size, :func:`box_blur`
         perform a box blur on the image data.
        """
        result = f.box_blur(a, size=2)
        assert (np.around(result, 4) == np.array([
            [0.2500, 0.2500, 0.5000, 0.7500, 0.8750],
            [0.2500, 0.2500, 0.5000, 0.7500, 0.8750],
            [0.5000, 0.5000, 0.7500, 0.8750, 0.7500],
            [0.7500, 0.7500, 0.8750, 0.7500, 0.5000],
            [0.8750, 0.8750, 0.7500, 0.5000, 0.2500],
        ], dtype=float)).all()

    def test_video(self, video_2_5_5):
        """Given three dimensional image data, :func:`box_blur`
        the blur should be performed on all frames of the image data.
        """
        result = f.box_blur(video_2_5_5, size=2)
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


class TestFilterGaussianBlue:
    def test_filter(self, a):
        """Given image data and a sigma, :func:`gaussian_blur`
        should perform a gaussian blur on the image data.
        """
        result = f.gaussian_blur(a, sigma=0.5)
        assert (np.around(result, 4) == np.array([
            [0.1070, 0.3036, 0.5534, 0.7918, 0.9158],
            [0.3036, 0.5002, 0.7442, 0.9046, 0.7918],
            [0.5534, 0.7442, 0.9044, 0.7442, 0.5534],
            [0.7918, 0.9046, 0.7442, 0.5002, 0.3036],
            [0.9158, 0.7918, 0.5534, 0.3036, 0.1070],
        ], dtype=float)).all()

    def test_video(self, video_2_5_5):
        """Given image data and a sigma, :func:`gaussian_blur`
        should perform a gaussian blur on the image data.
        """
        result = f.gaussian_blur(video_2_5_5, sigma=0.5)
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
        """Given image data and a size factor, :func:`glow`
        should zoom into the image by the size factor.
        """
        result = f.glow(video_2_5_5, sigma=4)
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


class TestFilterMotionBlur:
    def test_filter(self, a):
        """Given image data, an amount, and a direction,
        :func:`motion_blur` should perform a motion blur
        on the image data.
        """
        result = f.motion_blur(a, amount=2, axis=f.X_)
        assert (np.around(result, 4) == np.array([
            [0.1250, 0.1250, 0.3750, 0.6250, 0.8750],
            [0.3750, 0.3750, 0.6250, 0.8750, 0.8750],
            [0.6250, 0.6250, 0.8750, 0.8750, 0.6250],
            [0.8750, 0.8750, 0.8750, 0.6250, 0.3750],
            [0.8750, 0.8750, 0.6250, 0.3750, 0.1250],
        ], dtype=float)).all()

    def test_vertical(self, a):
        """Given image data, an amount, and a direction,
        :func:`motion_blur` should perform a motion blur
        on the image data. If the direction is the Y axis,
        the blur should be vertical.
        """
        result = f.motion_blur(a, amount=2, axis=f.Y_)
        assert (np.around(result, 4) == np.array([
            [0.1250, 0.3750, 0.6250, 0.8750, 0.8750],
            [0.1250, 0.3750, 0.6250, 0.8750, 0.8750],
            [0.3750, 0.6250, 0.8750, 0.8750, 0.6250],
            [0.6250, 0.8750, 0.8750, 0.6250, 0.3750],
            [0.8750, 0.8750, 0.6250, 0.3750, 0.1250],
        ], dtype=float)).all()

    def test_video(self, video_2_5_5):
        """Given image data, an amount, and a direction,
        :func:`motion_blur` should perform a motion blur
        on the video data.
        """
        result = f.motion_blur(video_2_5_5, amount=2, axis=f.X_)
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

    def test_invalid_axis(self, a):
        """If given an invalid axis, :func:`motion_blur` should
        raise a :class:`ValueError` exception.
        """
        with pt.raises(
            ValueError, match='motion_blur can only affect the X or Y axis.'
        ):
            _ = f.motion_blur(a, amount=2, axis=f.Z)
