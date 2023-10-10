"""
make_spacer
~~~~~~~~~~~

Create a black JPG that can be used as a spacer in a video.
"""
import argparse
from textwrap import dedent

import numpy as np

import pjimg.imgio as iw
from pjimg.util import RESOLUTIONS


def get_channels(color: str) -> tuple[int, int, int]:
    """Convert a 24-bit hex color string into an RGB color.

    :param color: A 24-bit hexadecimal color string, like is commonly
        used in HTML and CSS.
    :return: A :class:tuple object.
    :rtype: tuple
    """
    r = int(color[:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:], 16)
    return (r, g, b)


def main(filepath: str,
         res: tuple[int, int],
         color: tuple[int, int, int]) -> None:
    """Create a color image that can be used as a spacer in a video.

    :param filepath: The location to save the spacer image.
    :param res: The resolution of the image. This is a tuple of the
        form (x, y), where "x" is the width of the image in pixels and
        "y" is the height of the image in pixels.
    :param color: The color of the image. This is a tuple of integers
        representing the amounts of red, green, and blue (in that
        order) are contained in the color.
    :return: None.
    :rtype: None.
    """
    # Create the array of image data for the spacer image.
    a = np.zeros((1, *res[::-1], 3), dtype=int)
    for c in 0, 1, 2:
        a[:, :, :, c] = color[c]

    # Send that image and the save location to imagewriter.save_image.
    iw.write(filepath, a)


if __name__ == '__main__':
    # Define the command line options.
    resolutions = tuple(key for key in RESOLUTIONS)
    resolution_descr = '\n'.join(
        f'  * {key} ({RESOLUTIONS[key][0]}\u00d7{RESOLUTIONS[key][1]})'
        for key in RESOLUTIONS
    )
    options = {
        'filepath': {
            'args': ('filepath',),
            'kwargs': {
                'type': str,
                'action': 'store',
                'help': 'Where to save the spacer image.',
            }
        },
        'resolution': {
            'args': ('-r', '--resolution',),
            'kwargs': {
                'type': str,
                'action': 'store',
                'choices': resolutions,
                'help': 'The resolution of the video. See options below.',
                'metavar': 'RESOLUTION',
                'default': '720p'
            }
        },
        'color': {
            'args': ('-c', '--color',),
            'kwargs': {
                'type': str,
                'action': 'store',
                'help': 'The color of the frame in 24-bit hex.',
                'default': '000000'
            }
        },
    }

    # Read the command line arguments.
    p = argparse.ArgumentParser(
        prog='make_spacer.py',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Create a spacer image for video.',
        epilog=dedent('''\
        RESOLUTIONS
        -----------
        The following resolutions are available options:

        ''') + resolution_descr
    )
    for option in options:
        args = options[option]['args']
        kwargs = options[option]['kwargs']
        p.add_argument(*args, **kwargs)
    args = p.parse_args()

    # Create the spacer image.
    res = RESOLUTIONS[args.resolution]
    color = get_channels(args.color)
    main(args.filepath, res, color)
