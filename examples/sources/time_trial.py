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
from pjimg.imgio import write
from pjimg.util.model import Size


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)
logging.captureWarnings(True)


DEVICES = ['', 'cpu', 'mps']
# SIZE = (1, 2160, 3840)
# SIZE = (1, 1024, 768)
# SIZE = (1, 300, 300)
SIZE = (1, 120, 120)


def trial(src: srcs.Source, name: str, size: Size = SIZE) -> None:
    """Run a trial of an object."""
    t0 = dt.now()
    img = src.fill(size)
    write(name, img)
    duration = dt.now() - t0
    print(f'{duration} {name}')


def run_trial(cls, kwargs, ext, devices=DEVICES) -> None:
    if not ext:
        ext = 'jpg'
    else:
        ext = ext[0]
    name = cls.__name__.lower()
    title = f'{name} Trial'
    print(title)
    print('-' * len(title))
    for device in devices:
        kwargs['device'] = device
        src = cls(**kwargs)
        trial(src, f'{name}{device}.{ext}')
    print('-' * len(title))
    print()


if __name__ == '__main__':
    seed = 12345
    unit = (30, 30, 30)
    to_try = [
#         (srcs.Noise, {'seed': seed, 'device': '',}),
#         (srcs.Embers, {'depth': 8, 'seed': seed, 'device': '',}),
#         (srcs.CosineCurtains, {'unit': unit, 'seed': seed, 'device': '',}),
#         (srcs.CosineNoise, {'unit': unit, 'seed': seed, 'device': '',}),
#         (srcs.Curtains, {'unit': unit, 'seed': seed, 'device': '',}),
#         (srcs.UnitNoise, {'unit': unit, 'seed': seed, 'device': '',}),
#         (srcs.Perlin, {'unit': unit, 'seed': seed, 'device': '',}),
#         (srcs.OctaveUnitNoise, {'unit': unit, 'seed': seed, 'device': '',}),
#         (srcs.OctavePerlin, {'unit': unit, 'seed': seed, 'device': '',}),
#         (srcs.Maze, {'unit': (1, 30, 30), 'seed': seed, 'device': '',}),
#         (
#             srcs.AnimatedMaze,
#             {'unit': (1, 30, 30), 'delay': 4, 'linger': 5, 'seed': seed, 'device': '',},
#             'mp4'
#         ),
#         (srcs.SolvedMaze, {'unit': (1, 30, 30), 'seed': seed, 'device': ''}),
        (srcs.OctaveMaze, {'unit': (1, 30, 30), 'seed': seed, 'device': ''}),
    ]
    for cls, kwargs, *ext in to_try:
        run_trial(cls, kwargs, ext) #, ['cpu',])
