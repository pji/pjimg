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
        [srcs.Noise(seed=seed), 'noise.jpg'],
        [srcs.NoiseTorch(seed=seed), 'noisetorch.jpg'],
        [srcs.NoiseTorch(seed=seed, device='mps'), 'noisetorchmps.jpg'],
        
        # UnitNoises.
        [srcs.UnitNoise(unit, seed=seed), 'unitnoise.jpg'],
        [
            srcs.UnitNoiseTorch(unit, seed=seed, device='cpu'),
            'unitnoisetorch.jpg'
        ],
        [
            srcs.UnitNoiseTorch(unit, seed=seed, device='mps'),
            'unitnoisetorchmps.jpg'
        ],
        
        # Perlins.
        [srcs.Perlin(unit, seed=seed), 'perlin.jpg'],
        [srcs.PerlinTorch(unit, seed=seed), 'perlintorch.jpg'],
        [srcs.PerlinTorch(unit, seed=seed, device='mps'), 'perlintorchmps.jpg'],
        
        # OctaveUnitNoises.
        [srcs.OctaveUnitNoise(unit=unit, seed=seed), 'octaveunitnoise.jpg'],
        [
            srcs.OctaveUnitNoiseTorch(unit=unit, seed=seed, device='cpu'),
            'octaveunitnoisetorch.jpg'
        ],
        [
            srcs.OctaveUnitNoiseTorch(unit=unit, seed=seed, device='mps'),
            'octaveunitnoisetorchmps.jpg'
        ],
        
        # OctavePerlins.
        [srcs.OctavePerlin(unit=unit, seed=seed), 'octaveperlin.jpg'],
        [
            srcs.OctavePerlinTorch(unit=unit, seed=seed),
            'octaveperlintorch.jpg'],
        [
            srcs.OctavePerlinTorch(unit=unit, seed=seed, device='mps'),
            'octaveperlintorchmps.jpg'
        ],
    ]
    for args in to_try:
        trial(*args)
    