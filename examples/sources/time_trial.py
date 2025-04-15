"""
time_trial
~~~~~~~~~

A script to compare the generation times of different sources.
"""
import logging
import os
from datetime import datetime as dt

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from pjimg import sources as srcs
from pjimg.imgio import write_image
from pjimg.util.model import Size


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)
logging.captureWarnings(True)


SIZE = (1, 2160, 3840)


def trial(src: srcs.Source, name: str, size: Size = SIZE) -> None:
    """Run a trial of an object."""
    t0 = dt.now()
    img = src.fill(size)
    write_image(name, img)
    duration = dt.now() - t0
    print(f'{name:>35}: {duration}')


if __name__ == '__main__':
    seed = 12345
    unit = (SIZE[-1] // 8, SIZE[-1] // 8, SIZE[-1] // 8)
    to_try = [
        # Noises.
#         [srcs.Noise(seed=seed), 'noise.jpg'],
#         [srcs.Noise(seed=seed, device='cpu'), 'noisecpu.jpg'],
#         [srcs.Noise(seed=seed, device='mps'), 'noisemps.jpg'],
        
        # Embers.
#         [srcs.Embers(depth=8, seed=seed), 'embers.jpg'],
#         [srcs.Embers(depth=8, seed=seed, device='cpu'), 'emberscpu.jpg'],
#         [srcs.Embers(depth=8, seed=seed, device='mps'), 'embersmps.jpg'],
        
        # CosineNoise
#         [srcs.CosineNoise(unit, seed=seed), 'cosinenoise.jpg'],
#         [srcs.CosineNoise(unit, seed=seed, device='cpu'), 'cosinenoisecpu.jpg'],
#         [srcs.CosineNoise(unit, seed=seed, device='mps'), 'cosinenoisemps.jpg'],
        
        # UnitNoises.
        [srcs.UnitNoise(unit, seed=seed), 'unitnoise.jpg'],
        [srcs.UnitNoise(unit, seed=seed, device='cpu'), 'unitnoisecpu.jpg'],
        [srcs.UnitNoise(unit, seed=seed, device='mps'), 'unitnoisemps.jpg'],
        
        # Perlins.
#         [srcs.Perlin(unit, seed=seed), 'perlin.jpg'],
#         [srcs.Perlin(unit, seed=seed, device='cpu), 'perlincpu.jpg'],
#         [srcs.Perlin(unit, seed=seed, device='mps'), 'perlinmps.jpg'],
        
        # OctaveUnitNoises.
#         [srcs.OctaveUnitNoise(unit=unit, seed=seed), 'octaveunitnoise.jpg'],
#         [
#             srcs.OctaveUnitNoise(unit=unit, seed=seed, device='cpu'),
#             'octaveunitnoisecpu.jpg'
#         ],
#         [
#             srcs.OctaveUnitNoise(unit=unit, seed=seed, device='mps'),
#             'octaveunitnoisemps.jpg'
#         ],
        
        # OctavePerlins.
#         [srcs.OctavePerlin(unit=unit, seed=seed), 'octaveperlin.jpg'],
#         [
#             srcs.OctavePerlin(unit=unit, seed=seed, device='cpu'),
#             'octaveperlincpu.jpg'],
#         [
#             srcs.OctavePerlin(unit=unit, seed=seed, device='mps'),
#             'octaveperlinmps.jpg'
#         ],
    ]
    for args in to_try:
        trial(*args)
    