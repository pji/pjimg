"""
blender
~~~~~~~

A python script for blending image data.

for more information, view the help::

    python3 examples/blender.py -h

"""
from argparse import ArgumentParser
from inspect import getmembers, isfunction
from typing import Callable

import pjimg.imgio as iw
import pjimg.imgblend as ib


# Private utility functions.
def _get_blends() -> dict[str, Callable]:
    """Get the list of blending functions."""
    members = getmembers(ib, isfunction)
    blends = [blend for blend in members if not blend[0].startswith('can_')]
    blends = [blend for blend in blends if not blend[0].startswith('will_')]
    return dict(blends)


# Public functions.
def blend_images(
    file_a: str,
    file_b: str,
    blend: Callable,
    file_ab: str
) -> None:
    """Blend images."""
    # Load files.
    a = iw.read_image(file_a)
    b = iw.read_image(file_b)

    # Blend the image.
    ab = blend(a, b)

    # Save file.
    iw.write(file_ab, ab)


# Execution mainline.
def main() -> None:
    # Define the command line options.
    blends_ = _get_blends()
    options = {
        'existing': {
            'args': ('existing', ),
            'kwargs': {
                'type': str,
                'action': 'store',
                'help': 'The base file for the blend.',
            },
        },
        'blending': {
            'args': ('blending', ),
            'kwargs': {
                'type': str,
                'action': 'store',
                'help': 'The blending file for the blend.',
            },
        },
        'blend': {
            'args': ('blend', ),
            'kwargs': {
                'choices': blends_.keys(),
                'action': 'store',
                'help': 'The blend for the images.',
            },
        },
        'blended': {
            'args': ('blended', ),
            'kwargs': {
                'type': str,
                'action': 'store',
                'help': 'The blend for the images.',
            },
        },
    }

    # Read the command line arguments.
    p = ArgumentParser(
        prog='imgblender',
        description='Blend images or video.',
    )
    for option in options:
        args = options[option]['args']
        kwargs = options[option]['kwargs']
        p.add_argument(*args, **kwargs)
    args = p.parse_args()
    blend = blends_[args.blend]

    # Run the blend.
    blend_images(args.existing, args.blending, blend, args.blended)


if __name__ == '__main__':
    main()
