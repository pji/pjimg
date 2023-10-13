"""
test_examples
~~~~~~~~~~~~~

Unit tests for the examples for :mod:`pjimg.imgblend`.
"""
from pathlib import Path
from subprocess import run

import numpy as np
import pytest as pt


# Common functions.
def cmp_files(path_a, path_b):
    """Compare the contents of two files."""
    with open(path_a, 'rb') as fh:
        a = fh.read()
    with open(path_b, 'rb') as fh:
        b = fh.read()
    return a == b


# Fixtures.
@pt.fixture
def data_path():
    "The path for test data."
    return Path('tests/test_imgblend/data')


# Test cases.
class TestBlender:
    def test_blend(self, data_path, tmp_path):
        """Given the paths to two files, a blend function, and the
        path of a destination file, `blender.py` should run the blend
        on the contents of the two files and save the result in the
        destination file.
        """
        fname = '__test_examples_testblender_test_blend.jpg'
        run([
            'python', 'examples/imgblend/blender.py',
            data_path / '__test_horizontal_grayscale_image.jpg',
            data_path / '__test_vertical_grayscale_image.jpg',
            'multiply',
            tmp_path / fname,
        ])
        assert cmp_files(tmp_path / fname, data_path / fname)

    def test_blend_diff_size(self, data_path, tmp_path):
        """If the images are different sizes, `blender.py` should resize
        the smallest image to match the largest before blending.
        """
        fname = '__test_examples_testblender_test_blend_diff_size.jpg'
        run([
            'python', 'examples/imgblend/blender.py',
            data_path / '__test_5x5_grayscale_image.jpg',
            data_path / '__test_horizontal_grayscale_image.jpg',
            'multiply',
            tmp_path / fname,
        ])
        assert cmp_files(tmp_path / fname, data_path / fname)
