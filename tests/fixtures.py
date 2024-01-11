"""
fixtures
~~~~~~~~

Common test fixtures.
"""
import numpy as np
import pytest as pt


# Fixtures
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
