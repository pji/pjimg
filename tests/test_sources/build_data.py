"""
build_data
~~~~~~~~~~

Build the expected data files for the example tests.
"""
import pjimg.imgio as iw

import pjimg.sources as ig


def build_noisy():
    """Build the expected data for `examples/noisy.py`."""
    noise = ig.unitnoise.Curtains(
        unit=(1, 1024, 1024), seed='spam', repeats=1
    )
    a = noise.fill((1, 480, 640))
    iw.write('tests/test_sources/data/__test_noisy_curtains.jpg', a)

    noise = ig.unitnoise.CosineCurtains(
        unit=(1, 1024, 1024), seed='spam', repeats=1
    )
    a = noise.fill((1, 480, 640))
    iw.write('tests/test_sources/data/__test_noisy_coscurtains.jpg', a)

    noise = ig.maze.Maze(
        unit=(1, 20, 20), seed='spam', repeats=1
    )
    a = noise.fill((1, 480, 640))
    iw.write('tests/test_sources/data/__test_noisy_maze.jpg', a)

    noise = ig.noise.Noise(seed='spam')
    a = noise.fill((1, 480, 640))
    iw.write('tests/test_sources/data/__test_noisy_noise.jpg', a)

    noise = ig.unitnoise.OctaveCosineCurtains(
        unit=(1, 1024, 1024), seed='spam', repeats=1
    )
    a = noise.fill((1, 480, 640))
    iw.write('tests/test_sources/data/__test_noisy_ocoscurtains.jpg', a)

    noise = ig.unitnoise.OctaveCurtains(
        unit=(1, 1024, 1024), seed='spam', repeats=1
    )
    a = noise.fill((1, 480, 640))
    iw.write('tests/test_sources/data/__test_noisy_ocurtains.jpg', a)

    noise = ig.perlin.OctavePerlin(
        unit=(1, 1024, 1024), seed='spam', repeats=1
    )
    a = noise.fill((1, 480, 640))
    iw.write('tests/test_sources/data/__test_noisy_operlin.jpg', a)

    noise = ig.unitnoise.OctaveUnitNoise(
        unit=(1, 1024, 1024), seed='spam', repeats=1
    )
    a = noise.fill((1, 480, 640))
    iw.write('tests/test_sources/data/__test_noisy_ounitnoise.jpg', a)

    noise = ig.worley.OctaveWorley(points=16, seed='spam')
    a = noise.fill((1, 480, 640))
    iw.write('tests/test_sources/data/__test_noisy_oworley.jpg', a)

    noise = ig.perlin.Perlin(
        unit=(1, 1024, 1024), seed='spam', repeats=1
    )
    a = noise.fill((1, 480, 640))
    iw.write('tests/test_sources/data/__test_noisy_perlin.jpg', a)

    noise = ig.unitnoise.UnitNoise(
        unit=(1, 1024, 1024), seed='spam', repeats=1
    )
    a = noise.fill((1, 480, 640))
    iw.write('tests/test_sources/data/__test_noisy_unitnoise.jpg', a)

    noise = ig.worley.Worley(points=16, seed='spam')
    a = noise.fill((1, 480, 640))
    iw.write('tests/test_sources/data/__test_noisy_worley.jpg', a)


if __name__ == '__main__':
    build_noisy()
