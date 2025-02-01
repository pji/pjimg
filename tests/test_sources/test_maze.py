"""
test_maze
~~~~~~~~~

Unit tests for :mod:`pjimg.sources.maze`.
"""
import numpy as np
import pytest as pt

from pjimg.sources import maze as m
from tests.common import mkhex


class TestMaze:
    # Tests for Maze initiation.
    def test_init_all_default(self):
        """Given only required parameters, :class:`Maze` should
        initialize the required attributes with the given values. It
        should then initialize the optional attributes with default
        values.
        """
        required = {'unit': (4, 4, 4),}
        optional = {
            'width': 0.2,
            'inset': (0, 1, 1),
            'origin': 'tl',
            'min': 0x00,
            'max': 0xff,
            'repeats': 1,
            'seed': None,
        }
        obj = m.Maze(**required)
        for attr in required:
            assert getattr(obj, attr) == required[attr]
        for attr in optional:
            assert getattr(obj, attr) == optional[attr]

    def test_init_all_optional(self):
        """Given optional parameters, :class:`Maze` should
        initialize the given attributes with the given values.
        """
        required = {'unit': (4, 4, 4),}
        optional = {
            'width': 0.7,
            'inset': (0, 0, 0),
            'origin': (3, 3, 3),
            'min': 0x70,
            'max': 0x8f,
            'repeats': 3,
            'seed': 'spam',
        }
        obj = m.Maze(**required, **optional)
        for attr in required:
            assert getattr(obj, attr) == required[attr]
        for attr in optional:
            assert getattr(obj, attr) == optional[attr]

    # Tests for Maze.fill.
    def test_fill(self):
        """When given the size of an array, :meth:`Maze.fill` should
        return an array of that size filled with a maze.
        """
        maze = m.Maze(width=0.34, unit=(1, 3, 3), seed='spam')
        result = maze.fill((2, 10, 10))
        assert (mkhex(result) == np.array([
            [
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            ],
            [
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            ],
        ], dtype=np.uint8)).all()

    def test_fill_origin_br_zero_insert(self):
        """When given that the origin should be in the bottom-
        right of the fill, the maze's path should start being
        drawn from the bottom right of the fill.
        """
        maze = m.Maze(
            width=0.34,
            inset=(0, 0, 0),
            origin='br',
            unit=(1, 3, 3),
            seed='spam'
        )
        result = maze.fill((1, 9, 9))
        assert (mkhex(result) == np.array([
            [
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0xff],
                [0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0xff],
                [0x00, 0x00, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0xff],
                [0x00, 0x00, 0xff, 0xff, 0x00, 0xff, 0xff, 0xff, 0xff],
                [0x00, 0x00, 0xff, 0xff, 0x00, 0xff, 0xff, 0xff, 0xff],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff],
            ],
        ], dtype=np.uint8)).all()

    def test_fill_origin_mm_zero_insert(self):
        """When given that the origin should be in the middle of
        the fill, the maze's path should start being drawn from
        the bottom right of the fill.
        """
        maze = m.Maze(
            width=0.34,
            inset=(0, 0, 0),
            origin='mm',
            unit=(1, 3, 3),
            seed='spam'
        )
        result = maze.fill((1, 9, 9))
        assert (mkhex(result) == np.array([
            [
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0xff],
                [0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0xff],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff],
                [0x00, 0x00, 0xff, 0xff, 0x00, 0xff, 0xff, 0x00, 0xff],
                [0x00, 0x00, 0xff, 0xff, 0x00, 0xff, 0xff, 0x00, 0xff],
                [0x00, 0x00, 0xff, 0xff, 0x00, 0xff, 0xff, 0x00, 0xff],
                [0x00, 0x00, 0xff, 0xff, 0x00, 0xff, 0xff, 0xff, 0xff],
            ],
        ], dtype=np.uint8)).all()

    def test_fill_origin_br_zero_insert(self):
        """When given that the origin should be in the top-
        left of the fill, the maze's path should start being
        drawn from the bottom right of the fill.
        """
        maze = m.Maze(
            width=0.34,
            inset=(0, 0, 0),
            origin='tl',
            unit=(1, 3, 3),
            seed='spam'
        )
        result = maze.fill((1, 9, 9))
        assert (mkhex(result) == np.array([
            [
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff],
                [0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff],
                [0x00, 0x00, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0x00, 0xff, 0xff, 0xff, 0xff],
                [0x00, 0x00, 0xff, 0xff, 0x00, 0xff, 0xff, 0xff, 0xff],
                [0x00, 0x00, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0xff],
                [0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff],
            ],
        ], dtype=np.uint8)).all()


class TestAnimatedMaze:
    # Tests for initiation.
    def test_init_all_default(self):
        """Given only required parameters, :class:`AnimatedMaze` should
        initialize the required attributes with the given values. It
        should then initialize the optional attributes with default
        values.
        """
        required = {'unit': (4, 4, 4),}
        optional = {
            'delay': 0,
            'linger': 0,
            'trace': True,
            'width': 0.2,
            'inset': (0, 1, 1),
            'origin': 'tl',
            'min': 0x00,
            'max': 0xff,
            'repeats': 1,
            'seed': None,
        }
        obj = m.AnimatedMaze(**required)
        for attr in required:
            assert getattr(obj, attr) == required[attr]
        for attr in optional:
            assert getattr(obj, attr) == optional[attr]

    def test_init_all_optional(self):
        """Given optional parameters, :class:`AnimatedMaze` should
        initialize the given attributes with the given values.
        """
        required = {'unit': (4, 4, 4),}
        optional = {
            'delay': 1,
            'linger': 2,
            'trace': False,
            'width': 0.7,
            'inset': (0, 0, 0),
            'origin': (3, 3, 3),
            'min': 0x70,
            'max': 0x8f,
            'repeats': 3,
            'seed': 'spam',
        }
        obj = m.AnimatedMaze(**required, **optional)
        for attr in required:
            assert getattr(obj, attr) == required[attr]
        for attr in optional:
            assert getattr(obj, attr) == optional[attr]

    # Tests for fill.
    def test_fill(self):
        """When given the size of an array, :meth:`AnimatedMaze.fill`
        should return an array of that size filled with a maze.
        """
        maze = m.AnimatedMaze(
            width=0.34, inset=(0, 1, 1), unit=(1, 3, 3),
            origin='mm', seed='spam'
        )
        result = maze.fill((4, 9, 9))
        assert (mkhex(result) == np.array([
            [
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            ],
            [
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            ],
            [
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            ],
            [
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0x00, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0x00, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0x00, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            ],
        ], dtype=np.uint8)).all()

    def test_fill_with_delay(self):
        """If a delay is given, add that number of empty frames at
        the beginning of the image data.
        """
        maze = m.AnimatedMaze(
            delay=2, width=0.34, inset=(0, 1, 1), unit=(1, 3, 3),
            origin='mm', seed='spam'
        )
        result = maze.fill((4, 9, 9))
        assert (mkhex(result) == np.array([
            [
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            ],
            [
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            ],
            [
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            ],
            [
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            ],
            [
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            ],
            [
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0x00, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0x00, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0x00, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            ],
        ], dtype=np.uint8)).all()

    def test_fill_with_linger(self):
        """If a linger is given, add that number of copies of the
        last frame at the end of the image data.
        """
        maze = m.AnimatedMaze(
            linger=2, width=0.34, inset=(0, 1, 1), unit=(1, 3, 3),
            origin='mm', seed='spam'
        )
        result = maze.fill((4, 9, 9))
        assert (mkhex(result) == np.array([
            [
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            ],
            [
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            ],
            [
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            ],
            [
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0x00, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0x00, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0x00, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            ],
            [
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0x00, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0x00, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0x00, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            ],
            [
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0x00, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0x00, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0x00, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            ],
        ], dtype=np.uint8)).all()


class TestOctaveMaze:
    # Tests for Maze initiation.
    def test_init_all_default(self):
        """Given only required parameters, :class:`OctaveMaze` should
        initialize the required attributes with the given values. It
        should then initialize the optional attributes with default
        values.
        """
        required = {'unit': (4, 4, 4),}
        optional = {
            'width': 0.2,
            'inset': (0, 1, 1),
            'origin': 'tl',
            'min': 0x00,
            'max': 0xff,
            'repeats': 1,
            'seed': None,
        }
        obj = m.OctaveMaze(**required)
        for attr in required:
            assert getattr(obj, attr) == required[attr]
        for attr in optional:
            assert getattr(obj, attr) == optional[attr]

    def test_init_all_optional(self):
        """Given optional parameters, :class:`OctaveMaze` should
        initialize the given attributes with the given values.
        """
        required = {'unit': (4, 4, 4),}
        optional = {
            'width': 0.7,
            'inset': (0, 0, 0),
            'origin': (3, 3, 3),
            'min': 0x70,
            'max': 0x8f,
            'repeats': 3,
            'seed': 'spam',
        }
        obj = m.OctaveMaze(**required, **optional)
        for attr in required:
            assert getattr(obj, attr) == required[attr]
        for attr in optional:
            assert getattr(obj, attr) == optional[attr]

    # Tests for Maze.fill.
    def test_fill(self):
        """When given the size of an array, :meth:`OctaveMaze.fill` should
        return an array of that size filled with a maze.
        """
        maze = m.OctaveMaze(
            octaves=4,
            persistence=2,
            amplitude=2,
            frequency=3,
            unit=(1, 10, 10),
            width=0.34,
            seed='spam'
        )
        result = maze.fill((1, 20, 10))
        assert (mkhex(result) == np.array([
            [
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x19, 0x19, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x19, 0x19, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x19, 0x19, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x19, 0x19, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x19, 0x19, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x19, 0x19, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x19, 0x19, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x19, 0x19, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x19, 0x19, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x19, 0x19, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x19, 0x19, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x19, 0x19, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x19, 0x19, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x19, 0x19, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            ],
        ], dtype=np.uint8)).all()


class TestSolvedMaze:
    # Tests for initiation.
    def test_init_all_default(self):
        """Given only required parameters, :class:`SolvedMaze` should
        initialize the required attributes with the given values. It
        should then initialize the optional attributes with default
        values.
        """
        required = {'unit': (4, 4, 4),}
        optional = {
            'start': 'tl',
            'end': 'br',
            'algorithm': 'branches',
            'width': 0.2,
            'inset': (0, 1, 1),
            'origin': 'tl',
            'min': 0x00,
            'max': 0xff,
            'repeats': 1,
            'seed': None,
        }
        obj = m.SolvedMaze(**required)
        for attr in required:
            assert getattr(obj, attr) == required[attr]
        for attr in optional:
            assert getattr(obj, attr) == optional[attr]

    def test_init_all_optional(self):
        """Given optional parameters, :class:`AnimatedMaze` should
        initialize the given attributes with the given values.
        """
        required = {'unit': (4, 4, 4),}
        optional = {
            'delay': 1,
            'linger': 2,
            'trace': False,
            'width': 0.7,
            'inset': (0, 0, 0),
            'origin': (3, 3, 3),
            'min': 0x70,
            'max': 0x8f,
            'repeats': 3,
            'seed': 'spam',
        }
        obj = m.AnimatedMaze(**required, **optional)
        for attr in required:
            assert getattr(obj, attr) == required[attr]
        for attr in optional:
            assert getattr(obj, attr) == optional[attr]

    # Tests for fill.
    def test_fill(self):
        """When given the size of an array, :meth:`SolvedMaze.fill`
        should return an array of that size filled with a maze.
        """
        maze = m.SolvedMaze(width=0.34, unit=(1, 3, 3), seed='spam')
        result = maze.fill((1, 9, 9))
        assert (mkhex(result) == np.array([
            [
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            ],
        ], dtype=np.uint8)).all()

    def test_fill_end(self):
        """When given a end location, end the maze solution in
        that location in the maze.
        """
        maze = m.SolvedMaze(end='bl', width=0.34, unit=(1, 3, 3), seed='spam')
        result = maze.fill((1, 9, 9))
        assert (mkhex(result) == np.array([
            [
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            ],
        ], dtype=np.uint8)).all()

    def test_fill_start(self):
        """When given a start location, end the maze solution in
        that location in the maze.
        """
        maze = m.SolvedMaze(
            start='bl', width=0.34, unit=(1, 3, 3), seed='spam'
        )
        result = maze.fill((1, 9, 9))
        assert (mkhex(result) == np.array([
            [
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            ],
        ], dtype=np.uint8)).all()
