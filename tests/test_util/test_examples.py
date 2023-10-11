"""
test_examples
~~~~~~~~~~~~~

Unit tests to ensure the provided example code still works.
"""
from pathlib import Path
from subprocess import run

import pytest as pt


# Comman data.
data_path = Path('tests/test_util/data')


# Test utilities.
def compare_files(a, b):
    with open(a, 'rb') as fh:
        a_contents = fh.read()
    with open(b, 'rb') as fh:
        b_contents = fh.read()
    assert a_contents == b_contents


# Tests for resize_image.py.
def test_resize_image_magnify(tmp_path):
    """Given an image file, the save location, and a magnification
    factor, `resize_image.py` should save the resized image in the
    save location.
    """
    exp_file = data_path / '__test_resize_image_after_mag.jpg'
    src_file = data_path / '__test_resize_image_before.jpg'
    dst_file = tmp_path / 'spam.jpg'
    cmd = [
        'python',
        'examples/util/resize_image.py',
        src_file,
        dst_file,
        '-m', '10'
    ]
    run(cmd)
    compare_files(exp_file, dst_file)


def test_resize_image_resize(tmp_path):
    """Given an image file, the save location, and a new size,
    `resize_image.py` should save the resized image in the save
    location.
    """
    exp_file = data_path / '__test_resize_image_after.jpg'
    src_file = data_path / '__test_resize_image_before.jpg'
    dst_file = tmp_path / 'spam.jpg'
    cmd = [
        'python',
        'examples/util/resize_image.py',
        src_file,
        dst_file,
        '-s', '10', '10'
    ]
    run(cmd)
    compare_files(exp_file, dst_file)
