"""
build_doc_images
~~~~~~~~~~~~~~~~

Build the example images used in the documentation for :mod:`pjimg.imggen`.
"""
from argparse import ArgumentParser
from pathlib import Path

import pjimg.imggen as ig
from pjimg.imgio import write
from pjimg.util import Size, X, Y, Z


# Make the example images.
def make_patterns(size: Size, path: Path) -> None:
    """Create the example images for :mod:`pjimg.imggen.patterns`."""
    srcs = [
        ig.Box(
            origin=(n // 4 for n in size),
            dimensions=(1, *(n // 2 for n in size[Y:])),
            color=0.5
        ),
        ig.Gradient(
            direction='h',
            stops=(
                0.1, 0.0,
                0.2, 1.0,
                0.3, 0.0,
                0.8, 1.0,
                0.9, 0.0
            )
        ),
        ig.patterns.Hexes(radius=size[Y] // 8),
        ig.Lines(
            direction='v',
            length=(size[X] / 8) - (size[X] / 64)
        ),
        ig.Rays(count=7, offset=0.178),
        ig.Rings(
            radius=size[X] / 6,
            width=size[X] / 12,
            gap=size[X] / 18,
            count=6
        ),
        ig.Solid(color=0.25),
        ig.Spheres(radius=size[Y] / 16, offset=''),
        ig.Spot(radius=size[Y] * 2 / 3),
        ig.Text(
            text='SPAM',
            font='Helvetica',
            size=72,
            face=1,
            layout_engine='basic',
            origin=(size[X] / 2 - 107, size[Y] / 2 - 52),
            fill_color=0.75,
            bg_color=0.25,
            align='center',
            stroke_width=5,
            stroke_fill=0x00
        ),
        ig.Waves(length=size[Y] / 16, growth='g')
    ]
    for src in srcs:
        save(src, size, path)


def make_random(size: Size, path: Path) -> None:
    """Create the example images for random noise sources."""
    unit = (1, size[Y] // 5, size[Y] // 5)
    srcs = [
        ig.Noise(seed='spam'),
        ig.Embers(depth=6, seed='spam'),
        
#         ig.UnitNoise(
#             unit=unit,
#             seed='spam'
#         ),
#         ig.BorktaveCosineCurtains(
#             unit=unit,
#             octaves=3,
#             persistence=-4,
#             amplitude=24,
#             frequency=4,
#             seed='spam'
#         ),        
#         ig.Curtains(
#             unit=unit,
#             seed='spam'
#         ),
#         ig.CosineCurtains(
#             unit=unit,
#             seed='spam'
#         ),
#         ig.OctaveUnitNoise(
#             unit=(1, size[Y] * 9 // 2, size[Y] * 9 // 2),
#             octaves=6,
#             persistence=-4,
#             amplitude=48,
#             frequency=4,
#             seed='spam'
#         ),
#         ig.OctaveCurtains(
#             unit=unit,
#             octaves=3,
#             persistence=-4,
#             amplitude=24,
#             frequency=4,
#             seed='spam'
#         ),
#         ig.OctaveCosineCurtains(
#             unit=unit,
#             octaves=3,
#             persistence=-4,
#             amplitude=24,
#             frequency=4,
#             seed='spam'
#         ),
#         ig.Perlin(
#             unit=unit,
#             seed='spam'
#         ),
#         ig.OctavePerlin(
#             unit=(1, size[Y] * 9 // 2, size[Y] * 9 // 2),
#             octaves=6,
#             persistence=-4,
#             amplitude=24,
#             frequency=4,
#             seed='spam'
#         ),
#         ig.Maze(
#             unit=(1, size[Y] // 18, size[Y] // 18),
#             seed='spam'
#         ),
#         ig.SolvedMaze(
#             unit=(1, size[Y] // 18, size[Y] // 18),
#             seed='spam'
#         ),
#         ig.Worley(
#             points=20,
#             seed='spam'
#         ),
#         ig.OctaveWorley(
#             octaves=3,
#             persistence=6,
#             amplitude=5,
#             frequency=3,
#             points=8,
#             seed='spam'
#         ),
    ]
    for src in srcs:
        save(src, size, path)
    
#     vsrcs = [
#         ig.AnimatedMaze(
#             unit=(1, size[Y] // 18, size[Y] // 18),
#             width=0.34,
#             seed='spam'
#         ),
#     ]
#     for vsrc in vsrcs:
#         vsize = (250, size[Y], size[X])
#         save(vsrc, vsize, path, 'mp4')


def save(src: ig.Source, size: Size, path: Path, ext: str = 'jpg') -> None:
    """Generate the image data and save the file."""
    name = f'{type(src).__name__}.{ext}'.lower()
    print(f'Making {name}...', flush=True)
    a = src.fill(size)
    write(path / name, a)
    print(f'Made {name}.', flush=True)


# Mainline.
if __name__ == '__main__':
    p = ArgumentParser(
        description='Generate images for the documentation for imggen.',
        prog='imggen'
    )
    p.add_argument(
        '--outdir', '-o',
        action='store',
        default=Path('docs/source/images'),
        help='The directory to save the images.',
        type=Path
    )
    p.add_argument(
        '--size', '-s',
        action='store',
        default=(1280, 720),
        help='The height and width dimensions of the images.',
        nargs=2,
        type=int
    )
    args = p.parse_args()
    
    size = (1, args.size[1], args.size[0])
    path = args.outdir
    make_patterns(size, path)
    make_random(size, path)
