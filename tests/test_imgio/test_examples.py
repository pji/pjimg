"""
test_examples
~~~~~~~~~~~~~

Unit tests for the example scripts.
"""
from pathlib import Path
from subprocess import run


# Comman data.
data_path = Path('tests/test_imgio/data')


# Common test code.
def compare_files(a, b):
    """Compare the contents of two binary files."""
    with open(a, 'rb') as fh:
        a_content = fh.read()
    with open(b, 'rb') as fh:
        b_content = fh.read()
    assert a_content == b_content


# Tests for examples/make_color_fade.py.
def test_make_color_fade(tmp_path):
    """When called with a file path, a starting color, and an ending
    color, `make_color_fade.py` should create a video at the given
    location that fades from the starting color to the ending color.
    """
    expected = data_path / '__test_make_color_fade.mp4'
    path = tmp_path / '__test_make_color_fade.mp4'
    cmd = [
        'python',
        'examples/imgio/make_color_fade.py',
        path,
        '-s', '00ff00',
        '-e', 'ff00ff',
    ]
    run(cmd)
    compare_files(path, expected)


def test_make_color_fade_c(tmp_path):
    """When called with `-c` and a valid codec for the format,
    `make_color_fade.py` should create a video at the given
    location with the given frame rate and length.
    """
    expected = data_path / '__test_make_color_fade_c.mp4'
    path = tmp_path / '__test_make_color_fade_c.mp4'
    cmd = [
        'python',
        'examples/imgio/make_color_fade.py',
        path,
        '-s', '00ff00',
        '-e', 'ff00ff',
        '-c', 'hev1',
    ]
    run(cmd)
    compare_files(path, expected)


def test_make_color_fade_fl(tmp_path):
    """When called with `-f`, `-l`, and a valid resolution,
    `make_color_fade.py` should create a video at the given
    location with the given frame rate and length.
    """
    expected = data_path / '__test_make_color_fade_fl.mp4'
    path = tmp_path / '__test_make_color_fade_fl.mp4'
    cmd = [
        'python',
        'examples/imgio/make_color_fade.py',
        path,
        '-s', '00ff00',
        '-e', 'ff00ff',
        '-f', '12',
        '-l', '36',
    ]
    run(cmd)
    compare_files(path, expected)


def test_make_color_fade_r(tmp_path):
    """When called with `-r` and a valid resolution, `make_color_fade.py`
    should create a video at the given location with the given resolution.
    """
    expected = data_path / '__test_make_color_fade_r.mp4'
    path = tmp_path / '__test_make_color_fade_r.mp4'
    cmd = [
        'python',
        'examples/imgio/make_color_fade.py',
        path,
        '-s', '00ff00',
        '-e', 'ff00ff',
        '-r', 'dv_ntsc'
    ]
    run(cmd)
    compare_files(path, expected)


# Tests for examples/make_space.py.
def test_make_spacer(tmp_path):
    """When called with a file path, `make_spacer.py` should create
    an image at the given location that can be used as a video spacer.
    """
    expected = data_path / '__test_make_spacer.jpg'
    path = tmp_path / '__test_make_spacer.jpg'
    cmd = [
        'python',
        'examples/imgio/make_spacer.py',
        path,
    ]
    run(cmd)
    compare_files(path, expected)


def test_make_spacer_c(tmp_path):
    """When called with `-c`, `make_spacer.py` should create an image
    that is the given color.
    """
    expected = data_path / '__test_make_spacer_c.jpg'
    path = tmp_path / '__test_make_spacer_c.jpg'
    cmd = [
        'python',
        'examples/imgio/make_spacer.py',
        path,
        '-c', 'c05632',
    ]
    run(cmd)
    compare_files(path, expected)


def test_make_spacer_r(tmp_path):
    """When called with `-r`, `make_spacer.py` should create an image
    that is the given dimensions.
    """
    expected = data_path / '__test_make_spacer_r.jpg'
    path = tmp_path / '__test_make_spacer_r.jpg'
    cmd = [
        'python',
        'examples/imgio/make_spacer.py',
        path,
        '-r', 'dv_ntsc',
    ]
    run(cmd)
    compare_files(path, expected)
