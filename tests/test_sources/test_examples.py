"""
test_examples
~~~~~~~~~~~~~

Unit tests for the example scripts for :mod:`sources`.
"""
from pathlib import Path
from subprocess import run

import pytest as pt


# Common test code.
def compare_files(a, b):
    """Compare the contents of two binary files."""
    with open(a, 'rb') as fh:
        a_content = fh.read()
    with open(b, 'rb') as fh:
        b_content = fh.read()
    assert a_content == b_content


# Test cases.
@pt.mark.skip
class TestBuildDocImages:
    def test_p(self, tmp_path):
        """When invoked with `-o`, `build_doc_images` should write
        the documentation example images to the given directory.
        """
        exp_dir = Path('docs/source/images')
        fnames = [
            'animatedmaze.mp4',
            'box.jpg',
            'cosinecurtains.jpg',
            'curtains.jpg',
            'gradient.jpg',
            'lines.jpg',
            'maze.jpg',
            'noise.jpg',
            'octavecosinecurtains.jpg',
            'octavecurtains.jpg',
            'octaveperlin.jpg',
            'octaveunitnoise.jpg',
            'octaveworley.jpg',
            'perlin.jpg',
            'radials.jpg',
            'rays.jpg',
            'rings.jpg',
            'solid.jpg',
            'solvedmaze.jpg',
            'spheres.jpg',
            'spot.jpg',
            'text.jpg',
            'unitnoise.jpg',
            'worley.jpg',
        ]
        run([
            'python',
            'examples/sources/build_doc_images.py',
            '-o', str(tmp_path),
        ])
        for fname in fnames:
            compare_files(tmp_path / fname, exp_dir / fname)


class TestNoisy:
    def test_coscurtains(self, tmp_path):
        """When coscurtains is invoked, `noisy.py` should write one-
        dimensional cosine unit noise (cosine curtains) to the given
        file.
        """
        fname = '__test_noisy_coscurtains.jpg'
        expected = f'tests/test_sources/data/{fname}'
        actual = tmp_path / fname

        run([
            'python',
            'examples/sources/noisy.py',
            'coscurtains',
            '-s', 'spam',
            '640', '480',
            actual
        ])
        compare_files(actual, expected)

    def test_curtains(self, tmp_path):
        """When curtains is invoked, `noisy.py` should write one-
        dimensional unit noise (curtains) to the given file.
        """
        fname = '__test_noisy_curtains.jpg'
        expected = f'tests/test_sources/data/{fname}'
        actual = tmp_path / fname

        run([
            'python',
            'examples/sources/noisy.py',
            'curtains',
            '-s', 'spam',
            '640', '480',
            actual
        ])
        compare_files(actual, expected)

    def test_noise(self, tmp_path):
        """When noise is invoked, `noisy.py` should write random pixel
        noise to the given file.
        """
        fname = '__test_noisy_noise.jpg'
        expected = f'tests/test_sources/data/{fname}'
        actual = tmp_path / fname

        run([
            'python',
            'examples/sources/noisy.py',
            'noise',
            '-s', 'spam',
            '640', '480',
            actual
        ])
        compare_files(actual, expected)

    def test_coscurtains(self, tmp_path):
        """When ocoscurtains is invoked, `noisy.py` should write
        octave one-dimensional cosine unit noise (cosine curtains)
        to the given file.
        """
        fname = '__test_noisy_ocoscurtains.jpg'
        expected = f'tests/test_sources/data/{fname}'
        actual = tmp_path / fname

        run([
            'python',
            'examples/sources/noisy.py',
            'ocoscurtains',
            '-s', 'spam',
            '640', '480',
            actual
        ])
        compare_files(actual, expected)

    def test_maze(self, tmp_path):
        """When maze is invoked, `noisy.py` should write unit
        noise to the given file.
        """
        fname = '__test_noisy_maze.jpg'
        expected = f'tests/test_sources/data/{fname}'
        actual = tmp_path / fname

        run([
            'python',
            'examples/sources/noisy.py',
            'maze',
            '-s', 'spam',
            '640', '480',
            actual
        ])
        compare_files(actual, expected)

    def test_ocurtains(self, tmp_path):
        """When ocurtains is invoked, `noisy.py` should write octave
        one-dimensional unit noise (curtains) to the given file.
        """
        fname = '__test_noisy_ocurtains.jpg'
        expected = f'tests/test_sources/data/{fname}'
        actual = tmp_path / fname

        run([
            'python',
            'examples/sources/noisy.py',
            'ocurtains',
            '-s', 'spam',
            '640', '480',
            actual
        ])
        compare_files(actual, expected)

    def test_operlin(self, tmp_path):
        """When operlin is invoked, `noisy.py` should write octave
        perlin noise to the given file.
        """
        fname = '__test_noisy_operlin.jpg'
        expected = f'tests/test_sources/data/{fname}'
        actual = tmp_path / fname

        run([
            'python',
            'examples/sources/noisy.py',
            'operlin',
            '-s', 'spam',
            '640', '480',
            actual
        ])
        compare_files(actual, expected)

    def test_ounitnoise(self, tmp_path):
        """When ounitnoise is invoked, `noisy.py` should write octave
        unit noise to the given file.
        """
        fname = '__test_noisy_ounitnoise.jpg'
        expected = f'tests/test_sources/data/{fname}'
        actual = tmp_path / fname

        run([
            'python',
            'examples/sources/noisy.py',
            'ounitnoise',
            '-s', 'spam',
            '640', '480',
            actual
        ])
        compare_files(actual, expected)

    def test_perlin(self, tmp_path):
        """When perlin is invoked, `noisy.py` should write perlin
        noise to the given file.
        """
        fname = '__test_noisy_perlin.jpg'
        expected = f'tests/test_sources/data/{fname}'
        actual = tmp_path / fname

        run([
            'python',
            'examples/sources/noisy.py',
            'perlin',
            '-s', 'spam',
            '640', '480',
            actual
        ])
        compare_files(actual, expected)

    def test_unitnoise(self, tmp_path):
        """When unitnoise is invoked, `noisy.py` should write unit
        noise to the given file.
        """
        fname = '__test_noisy_unitnoise.jpg'
        expected = f'tests/test_sources/data/{fname}'
        actual = tmp_path / fname

        run([
            'python',
            'examples/sources/noisy.py',
            'unitnoise',
            '-s', 'spam',
            '640', '480',
            actual
        ])
        compare_files(actual, expected)

    def test_worley(self, tmp_path):
        """When worley is invoked, `noisy.py` should write worley
        noise to the given file.
        """
        fname = '__test_noisy_worley.jpg'
        expected = f'tests/test_sources/data/{fname}'
        actual = tmp_path / fname

        run([
            'python',
            'examples/sources/noisy.py',
            'worley',
            '-s', 'spam',
            '640', '480',
            actual
        ])
        compare_files(actual, expected)
