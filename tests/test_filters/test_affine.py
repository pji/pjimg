"""
test_affine
~~~~~~~~~~~

Unit tests for :mod:`pjimg.filters.affine`.
"""
import numpy as np
import pytest as pt

import pjimg.filters.affine as f
from tests.fixtures import *


# Test cases.
class TestFlip:
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


class TestRotate2d:
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


class TestRotate90:
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


class TestSkew:
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
