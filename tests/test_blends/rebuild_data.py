"""
rebuild_data
~~~~~~~~~~~~

A scrip to rebuild the test data for :mod:`blendser`.
"""
from pathlib import Path
from subprocess import run

import numpy as np
import pjimg.imgio as iw


# Locations.
data_path = Path('tests/test_blends/data')

# Build examples/blender.py data.
gs_5x5 = np.array([[
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.5, 1.0, 0.0],
    [0.0, 0.0, 0.5, 1.0, 0.0],
    [0.0, 0.0, 0.5, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0],
]], dtype=float)
iw.write(data_path / '__test_5x5_grayscale_image.jpg', gs_5x5)
gs_horiz_3x3 = np.array([[
    [0.0, 0.0, 0.0],
    [0.5, 0.5, 0.5],
    [1.0, 1.0, 1.0],
]], dtype=float)
iw.write(
    data_path / '__test_horizontal_grayscale_image.jpg',
    gs_horiz_3x3
)
gs_vert_3x3 = np.array([[
    [0.0, 0.5, 1.0],
    [0.0, 0.5, 1.0],
    [0.0, 0.5, 1.0],
]], dtype=float)
iw.write(
    data_path / '__test_vertical_grayscale_image.jpg', gs_vert_3x3
)
run([
    'python', 'examples/blends/blender.py',
    data_path / '__test_horizontal_grayscale_image.jpg',
    data_path / '__test_vertical_grayscale_image.jpg',
    'multiply',
    data_path / '__test_examples_testblender_test_blend.jpg'
])
run([
    'python', 'examples/blends/blender.py',
    data_path / '__test_5x5_grayscale_image.jpg',
    data_path / '__test_horizontal_grayscale_image.jpg',
    'multiply',
    data_path / '__test_examples_testblender_test_blend_diff_size.jpg'
])
