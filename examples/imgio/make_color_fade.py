"""
make_color_fade
~~~~~~~~~~~~~~~

Create a video that fades from one color to another.
"""
import argparse
from textwrap import dedent

import numpy as np

import pjimg.imgio as iw
from pjimg.util import RESOLUTIONS


# Constants.
R, G, B = 0, 1, 2
SUPPORTED = iw.VALID_FORMATS


# Utility functions.
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


def build_formats_description() -> str:
    """Build a list of the supported formats for the help file."""
    lables = ('FORMAT', 'CODECS',)
    title = 'SUPPORTED FILE FORMATS'
    descr = 'The following container formats are known to be supported.'
    tbl_tmp = '    {:<11} {}'
    formats = [
        tbl_tmp.format(vid, ','.join(SUPPORTED[vid].codecs))
        for vid in SUPPORTED
        if isinstance(SUPPORTED[vid], iw.Video)
    ]
    return '\n'.join((
        title,
        '-' * len(title),
        descr,
        '',
        tbl_tmp.format(*lables),
        *formats,
        ''
    ))


# Mainline.
def main(
    filepath: str,
    res: tuple[int, int],
    start_color: tuple[int, int, int],
    end_color: tuple[int, int, int],
    frames: int,
    framerate: float,
    codec: str = 'mp4v'
) -> None:
    """Create a video that fades from one color to another.

    :param filepath: The location to save the spacer image.
    :param res: The resolution of the image. This is a tuple of the
        form (x, y), where "x" is the width of the image in pixels and
        "y" is the height of the image in pixels.
    :param start_color: This is a tuple of integers
        representing the amounts of red, green, and blue (in that
        order) are contained in the starting color of the fade.
    :param end_color: This is a tuple of integers
        representing the amounts of red, green, and blue (in that
        order) are contained in the final color of the fade.
    :param frames: The number of frames for the transition.
    :param framerate: The frame rate of the video.
    :return: None.
    :rtype: None.
    """
    # Create the array of image data for the fade.
    diff_inc = [-1 * (s - e) / frames for s, e in zip(start_color, end_color)]
    a = np.indices((frames, *res[::-1], 3), dtype=np.float32)[0]
    for c in R, G, B:
        a[:, :, :, c] *= diff_inc[c]
        a[:, :, :, c] += start_color[c]
    a = a.astype(np.uint8)

    # Send that image and the save location to imagewriter.save_video.
    iw.write_video(filepath, a, framerate, codec=codec)


if __name__ == '__main__':
    # Define the command line options.
    resolutions = tuple(key for key in RESOLUTIONS)
    resolution_descr = '\n'.join(
        f'  * {key} ({RESOLUTIONS[key][0]}\u00d7{RESOLUTIONS[key][1]})'
        for key in RESOLUTIONS
    )
    options = {
        'codec': {
            'args': ('-c', '--codec'),
            'kwargs': {
                'type': str,
                'action': 'store',
                'help': 'The codec to encode the video.',
                'default': 'mp4v',
            },
        },
        'end_color': {
            'args': ('-e', '--end_color',),
            'kwargs': {
                'type': str,
                'action': 'store',
                'help': 'The color of the last frame in 24-bit hex.',
                'default': '000000'
            }
        },
        'filepath': {
            'args': ('filepath',),
            'kwargs': {
                'type': str,
                'action': 'store',
                'help': 'Where to save the spacer image.',
            }
        },
        'framerate': {
            'args': ('-f', '--framerate'),
            'kwargs': {
                'type': int,
                'action': 'store',
                'help': 'The frame rate of the fade.',
                'default': 24,
            }
        },
        'length': {
            'args': ('-l', '--length'),
            'kwargs': {
                'type': int,
                'action': 'store',
                'help': 'The number of frames for the fade.',
                'default': 72,
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
        'start_color': {
            'args': ('-s', '--start_color',),
            'kwargs': {
                'type': str,
                'action': 'store',
                'help': 'The color of the first frame in 24-bit hex.',
                'default': 'ffffff'
            }
        },
    }

    # Read the command line arguments.
    p = argparse.ArgumentParser(
        prog='make_color_fade.py',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Create a video of a color fade.',
        epilog=(
            dedent('''\
            RESOLUTIONS
            -----------
            The following resolutions are available options:

            ''')
            + resolution_descr
            + '\n\n'
            + build_formats_description()
        )
    )
    for option in options:
        args = options[option]['args']
        kwargs = options[option]['kwargs']
        p.add_argument(*args, **kwargs)
    args = p.parse_args()

    # Create the spacer image.
    res = RESOLUTIONS[args.resolution]
    start_color = get_channels(args.start_color)
    end_color = get_channels(args.end_color)
    main(
        args.filepath,
        res,
        start_color,
        end_color,
        args.length,
        args.framerate,
        args.codec
    )
