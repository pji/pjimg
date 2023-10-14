"""
build_doc_images
~~~~~~~~~~~~~~~~

Create the images used in the documentation for :mod:`pjimg.imgease`.
"""
from argparse import ArgumentParser
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

import pjimg.imgease as ie
import pjimg.imggen as ig
import pjimg.imgio as iw


# Make example images.
def make_images(path: Path, size: Sequence[int]) -> None:
    """Create the curves for the eases."""
    for ease in ie.eases:
        print(f'Building {ease}...')
        make_curve(path, ie.eases[ease])
        make_example(path, size, ie.eases[ease])
        print(f'{ease} built.')


def make_curve(path: Path, ease: ie.Ease) -> None:
    """Create the curve for an ease."""
    # Create the curve data.
    base = np.arange(129, dtype=float) / 128
    eased = ease(base.copy())
    
    # Plot the curve.
    plt.style.use('dark_background')
    fig, ax = plt.subplots()
    ax.plot(base, eased)

    # Save the file and close the plot.
    fname = f'plot_{ease.__name__}.png'
    fig.savefig(path / fname, bbox_inches="tight")
    plt.close()


def make_example(path: Path, size: Sequence[int], ease: ie.Ease) -> None:
    """Create an example image for the ease."""
    X, Y, Z = 2, 1, 0
    a = np.arange(size[X], dtype=float) / 1279
    a = np.tile(a[np.newaxis, np.newaxis, ...], (1, size[Y], 1))
    a[:, size[Y] // 2:, ...] = ease(a[:, size[Y] // 2:, ...])
    a[a > 1.0] = 1.0
    a[a < 0.0] = 0.0
    
    height = int(size[Y] * 1.1 // 12)
    origin = (size[X] // 10, height // 20)
    text_orig = ig.Text(
        '\u25b2 ORIGINAL \u25b2',
        font='Menlo',
        size=size[Y] // 30,
        origin=origin
    )
    height = int(size[Y] * 1.1 // 24)
    label = text_orig.fill((size[Z], height, size[X]))
    
    height = int(size[Y] * 1.1 // 12)
    origin = (size[X] * 8 // 10, height // 20)
    text_ease = ig.Text(
        '\u25bc EASED \u25bc',
        font='Menlo',
        size=size[Y] // 30,
        origin=origin
    )
    height = int(size[Y] * 1.1 // 24)
    label += text_ease.fill((size[Z], height, size[X]))
    label[label > 1.0] = 1.0
    label[label < 0.0] = 0.0
    
    ystart = size[Y] // 2 - height // 2
    ystop = ystart + height
    a[:, ystart:ystop, :] = label
    
    fname = f'ex_{ease.__name__}.png'
    iw.write(path / fname, a)


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
    make_images(path, size)
