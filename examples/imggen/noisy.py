"""
noisy
~~~~~

Generate an image containing visual noise.
"""
import argparse as ap
from pathlib import Path
from typing import Union

import pjimg.imggen as imggen
from pjimg.imgio import write
from pjimg.util import ImgAry


# Defaults.
OctaveNoiseDefaults = imggen.unitnoise.OctaveNoiseDefaults


# Noise generators.
def make_coscurtains(args: ap.Namespace) -> ImgAry:
    """Generate cosine curtains.
    
    :param args: The arguments passed from the command line.
    :return: The noise as a :class:`numpy.ndarray`.
    :rtype: numpy.ndarray
    """
    noise = imggen.CosineCurtains(
        unit=(1, *args.unit),
        min=args.min,
        max=args.max,
        repeats=args.repeats,
        seed=args.seed
    )
    size = (1, args.height, args.width)
    loc = (0, *args.location)
    return noise.fill(size, loc)


def make_curtains(args: ap.Namespace) -> ImgAry:
    """Generate curtains.
    
    :param args: The arguments passed from the command line.
    :return: The noise as a :class:`numpy.ndarray`.
    :rtype: numpy.ndarray
    """
    noise = imggen.Curtains(
        unit=(1, *args.unit),
        min=args.min,
        max=args.max,
        repeats=args.repeats,
        seed=args.seed
    )
    size = (1, args.height, args.width)
    loc = (0, *args.location)
    return noise.fill(size, loc)


def make_maze(args: ap.Namespace) -> ImgAry:
    """Generate a maze.
    
    :param args: The arguments passed from the command line.
    :return: The noise as a :class:`numpy.ndarray`.
    :rtype: numpy.ndarray
    """
    noise = imggen.Maze(
        unit=(1, *args.unit),
        width=args.path_width,
        inset=(0, *args.inset),
        origin=args.origin,
        min=args.min,
        max=args.max,
        repeats=args.repeats,
        seed=args.seed
    )
    size = (1, args.height, args.width)
    loc = (0, *args.location)
    return noise.fill(size, loc)


def make_noise(args: ap.Namespace) -> ImgAry:
    """Generate random pixel value noise.
    
    :param args: The arguments passed from the command line.
    :return: The noise as a :class:`numpy.ndarray`.
    :rtype: numpy.ndarray
    """
    noise = imggen.Noise(seed=args.seed)
    size = (1, args.height, args.width)
    loc = (0, *args.location)
    return noise.fill(size, loc)


def make_ocoscurtains(args: ap.Namespace) -> ImgAry:
    """Generate octaves cosine curtains.
    
    :param args: The arguments passed from the command line.
    :return: The noise as a :class:`numpy.ndarray`.
    :rtype: numpy.ndarray
    """
    noise = imggen.OctaveCosineCurtains(
        octaves=args.octaves,
        persistence=args.persistence,
        amplitude=args.amplitude,
        frequency=args.frequency,
        unit=(1, *args.unit),
        min=args.min,
        max=args.max,
        repeats=args.repeats,
        seed=args.seed
    )
    size = (1, args.height, args.width)
    loc = (0, *args.location)
    return noise.fill(size, loc)


def make_ocurtains(args: ap.Namespace) -> ImgAry:
    """Generate octave curtains.
    
    :param args: The arguments passed from the command line.
    :return: The noise as a :class:`numpy.ndarray`.
    :rtype: numpy.ndarray
    """
    noise = imggen.OctaveCurtains(
        octaves=args.octaves,
        persistence=args.persistence,
        amplitude=args.amplitude,
        frequency=args.frequency,
        unit=(1, *args.unit),
        min=args.min,
        max=args.max,
        repeats=args.repeats,
        seed=args.seed
    )
    size = (1, args.height, args.width)
    loc = (0, *args.location)
    return noise.fill(size, loc)


def make_operlin(args: ap.Namespace) -> ImgAry:
    """Generate octave perlin noise.
    
    :param args: The arguments passed from the command line.
    :return: The noise as a :class:`numpy.ndarray`.
    :rtype: numpy.ndarray
    """
    noise = imggen.OctavePerlin(
        octaves=args.octaves,
        persistence=args.persistence,
        amplitude=args.amplitude,
        frequency=args.frequency,
        unit=(1, *args.unit),
        min=args.min,
        max=args.max,
        repeats=args.repeats,
        seed=args.seed
    )
    size = (1, args.height, args.width)
    loc = (0, *args.location)
    return noise.fill(size, loc)


def make_ounitnoise(args: ap.Namespace) -> ImgAry:
    """Generate octave random blob noise.
    
    :param args: The arguments passed from the command line.
    :return: The noise as a :class:`numpy.ndarray`.
    :rtype: numpy.ndarray
    """
    noise = imggen.OctaveUnitNoise(
        octaves=args.octaves,
        persistence=args.persistence,
        amplitude=args.amplitude,
        frequency=args.frequency,
        unit=(1, *args.unit),
        min=args.min,
        max=args.max,
        repeats=args.repeats,
        seed=args.seed
    )
    size = (1, args.height, args.width)
    loc = (0, *args.location)
    return noise.fill(size, loc)


def make_oworley(args: ap.Namespace) -> ImgAry:
    """Generate octave worley noise.
    
    :param args: The arguments passed from the command line.
    :return: The noise as a :class:`numpy.ndarray`.
    :rtype: numpy.ndarray
    """
    vwidth, vheight = args.volume
    if vwidth < 0:
        vwidth = args.width
    if vheight < 0:
        vheight = args.height
        
    noise = imggen.OctaveWorley(
        octaves=args.octaves,
        persistence=args.persistence,
        amplitude=args.amplitude,
        frequency=args.frequency,
        points=args.points,
        volume=(1, vheight, vwidth),
        origin=(0, *args.origin),
        seed=args.seed
    )
    size = (1, args.height, args.width)
    loc = (0, *args.location)
    return noise.fill(size, loc)


def make_perlin(args: ap.Namespace) -> ImgAry:
    """Generate perlin noise.
    
    :param args: The arguments passed from the command line.
    :return: The noise as a :class:`numpy.ndarray`.
    :rtype: numpy.ndarray
    """
    noise = imggen.Perlin(
        unit=(1, *args.unit),
        min=args.min,
        max=args.max,
        repeats=args.repeats,
        seed=args.seed
    )
    size = (1, args.height, args.width)
    loc = (0, *args.location)
    return noise.fill(size, loc)


def make_unitnoise(args: ap.Namespace) -> ImgAry:
    """Generate random blob noise.
    
    :param args: The arguments passed from the command line.
    :return: The noise as a :class:`numpy.ndarray`.
    :rtype: numpy.ndarray
    """
    noise = imggen.UnitNoise(
        unit=(1, *args.unit),
        min=args.min,
        max=args.max,
        repeats=args.repeats,
        seed=args.seed
    )
    size = (1, args.height, args.width)
    loc = (0, *args.location)
    return noise.fill(size, loc)


def make_worley(args: ap.Namespace) -> ImgAry:
    """Generate random blob noise.
    
    :param args: The arguments passed from the command line.
    :return: The noise as a :class:`numpy.ndarray`.
    :rtype: numpy.ndarray
    """
    vwidth, vheight = args.volume
    if vwidth < 0:
        vwidth = args.width
    if vheight < 0:
        vheight = args.height
        
    noise = imggen.Worley(
        points=args.points,
        volume=(1, vheight, vwidth),
        origin=(0, *args.origin),
        seed=args.seed
    )
    size = (1, args.height, args.width)
    loc = (0, *args.location)
    return noise.fill(size, loc)


# Pattern generators.
def make_spheres(args: ap.Namespace) -> ImgAry:
    """Generate spheres.
    
    :param args: The arguments passed from the command line.
    :return: The noise as a :class:`numpy.ndarray`.
    :rtype: numpy.ndarray
    """
    noise = imggen.Spheres(radius=args.radius, offset=args.offset)
    size = (1, args.height, args.width)
    loc = (0, *args.location)
    return noise.fill(size, loc)


# Command line interface.
def parse_invocation() -> ap.ArgumentParser:
    """Build a CLI parser."""
    p = ap.ArgumentParser(
        description='Generate an image containing visual noise.',
        prog='noisy'
    )
    spa = p.add_subparsers(help='The type of noise.', required=True)
    
    # Noise
    parse_coscurtains(spa)
    parse_curtains(spa)
    parse_maze(spa)
    parse_noise(spa)
    parse_ocoscurtains(spa)
    parse_ocurtains(spa)
    parse_operlin(spa)
    parse_ounitnoise(spa)
    parse_oworley(spa)
    parse_perlin(spa)
    parse_unitnoise(spa)
    parse_worley(spa)
    
    # Patterns
    parse_spheres(spa)
    
    p.add_argument(
        'width',
        action='store',
        help='The width of the image.',
        type=int
    )
    p.add_argument(
        'height',
        action='store',
        help='The height of the image.',
        type=int
    )
    p.add_argument(
        'path',
        action='store',
        help='Where to save the image.',
        type=str
    )
    p.add_argument(
        '-l', '--location',
        action='store',
        default=(0, 0),
        help='Location within the noise to sample for image.',
        nargs=2,
        type=int
    )
    
    return p.parse_args()


# Create the subparser for each type of noise.
def parse_coscurtains(spa: ap._SubParsersAction) -> None:
    """Parse arguments for coscurtains.
    
    :param spa: The subparser.
    :return: None.
    :rtype: NoneType
    """
    sp = spa.add_parser(
        'coscurtains',
        description='Generate cosine curtains.'
    )
    add_unitnoise_arguments(sp)
    add_noise_arguments(sp)
    sp.set_defaults(func=make_coscurtains)


def parse_curtains(spa: ap._SubParsersAction) -> None:
    """Parse arguments for curtains.
    
    :param spa: The subparser.
    :return: None.
    :rtype: NoneType
    """
    sp = spa.add_parser(
        'curtains',
        description='Generate curtains.'
    )
    add_unitnoise_arguments(sp)
    add_noise_arguments(sp)
    sp.set_defaults(func=make_curtains)


def parse_maze(spa: ap._SubParsersAction) -> None:
    """Parse arguments for mazes.
    
    :param spa: The subparser.
    :return: None.
    :rtype: NoneType
    """
    sp = spa.add_parser(
        'maze',
        description='Generate a maze.'
    )
    add_maze_arguments(sp)
    add_noise_arguments(sp)
    sp.set_defaults(func=make_maze)


def parse_noise(spa: ap._SubParsersAction) -> None:
    """Parser for noise.
    
    :param spa: The subparser.
    :return: None.
    :rtype: NoneType
    """
    sp = spa.add_parser(
        'noise',
        description='Generate randomized pixel noise.'
    )
    add_noise_arguments(sp)
    sp.set_defaults(func=make_noise)


def parse_ocoscurtains(spa: ap._SubParsersAction) -> None:
    """Parse arguments for ocoscurtains.
    
    :param spa: The subparser.
    :return: None.
    :rtype: NoneType
    """
    sp = spa.add_parser(
        'ocoscurtains',
        description='Generate octave cosine curtains.'
    )
    add_octave_arguments(sp)
    add_unitnoise_arguments(sp)
    add_noise_arguments(sp)
    sp.set_defaults(func=make_ocoscurtains)


def parse_ocurtains(spa: ap._SubParsersAction) -> None:
    """Parse arguments for octave curtains.
    
    :param spa: The subparser.
    :return: None.
    :rtype: NoneType
    """
    sp = spa.add_parser(
        'ocurtains',
        description='Generate octave curtains.'
    )
    add_octave_arguments(sp)
    add_unitnoise_arguments(sp)
    add_noise_arguments(sp)
    sp.set_defaults(func=make_ocurtains)


def parse_operlin(spa: ap._SubParsersAction) -> None:
    """Parse arguments for operlin.
    
    :param spa: The subparser.
    :return: None.
    :rtype: NoneType
    """
    sp = spa.add_parser(
        'operlin',
        description='Generate octave perlin noise.'
    )
    add_octave_arguments(sp, imggen.perlin.defaults)
    add_unitnoise_arguments(sp)
    add_noise_arguments(sp)
    sp.set_defaults(func=make_operlin)


def parse_ounitnoise(spa: ap._SubParsersAction) -> None:
    """Parse arguments for octave unit noise.
    
    :param spa: The subparser.
    :return: None.
    :rtype: NoneType
    """
    sp = spa.add_parser(
        'ounitnoise',
        description='Generate unit noise.'
    )
    add_unitnoise_arguments(sp)
    add_octave_arguments(sp)
    add_noise_arguments(sp)
    sp.set_defaults(func=make_ounitnoise)


def parse_oworley(spa: ap._SubParsersAction) -> None:
    """Parse arguments for octave worley noise.
    
    :param spa: The subparser.
    :return: None.
    :rtype: NoneType
    """
    sp = spa.add_parser(
        'oworley',
        description='Generate octave worley noise.'
    )
    add_octave_arguments(sp)
    add_worley_arguments(sp)
    add_noise_arguments(sp)
    sp.set_defaults(func=make_worley)


def parse_perlin(spa: ap._SubParsersAction) -> None:
    """Parse arguments for perlin.
    
    :param spa: The subparser.
    :return: None.
    :rtype: NoneType
    """
    sp = spa.add_parser(
        'perlin',
        description='Generate perlin noise.'
    )
    add_unitnoise_arguments(sp)
    add_noise_arguments(sp)
    sp.set_defaults(func=make_perlin)


def parse_unitnoise(spa: ap._SubParsersAction) -> None:
    """Parse arguments for unit noise.
    
    :param spa: The subparser.
    :return: None.
    :rtype: NoneType
    """
    sp = spa.add_parser(
        'unitnoise',
        description='Generate unit noise.'
    )
    add_unitnoise_arguments(sp)
    add_noise_arguments(sp)
    sp.set_defaults(func=make_unitnoise)


def parse_worley(spa: ap._SubParsersAction) -> None:
    """Parse arguments for worley noise.
    
    :param spa: The subparser.
    :return: None.
    :rtype: NoneType
    """
    sp = spa.add_parser(
        'worley',
        description='Generate worley noise.'
    )
    add_worley_arguments(sp)
    add_noise_arguments(sp)
    sp.set_defaults(func=make_worley)


# Create the subparser for each type of pattern.
def parse_spheres(spa: ap._SubParsersAction) -> None:
    """Parser for spheres.
    
    :param spa: The subparser.
    :return: None.
    :rtype: NoneType
    """
    sp = spa.add_parser(
        'spheres',
        description='Generate spheres.'
    )
    sp.add_argument(
        '-F', '--offset',
        action='store',
        default='',
        help='The axis to offset.',
        type=str
    )
    sp.add_argument(
        '-r', '--radius',
        action='store',
        default=50.0,
        help='The radius of each sphere.',
        type=float
    )
    sp.set_defaults(func=make_spheres)


# Create the arguments for the subparsers. Each subparser is tied to
# one type of noise. Since many of the types of noise are subclasses
# of each other, many of them have the same parameters. So, many of
# the subparsers can share the same arguments.
def add_maze_arguments(sp: ap.ArgumentParser) -> None:
    """Add maze arguments to a subparser. This has to cover the unit
    noise arguments too, since :class:`imggen.Maze` needs a different
    default value for :attr:`imggen.UnitNoise.unit`.
    
    :param sp: A subparser that accepts noise arguments.
    :return: None.
    :rtype: NoneType
    """
    sp.add_argument(
        '-u', '--unit',
        action='store',
        default=(20, 20),
        help='The size of a unit within the noise.',
        nargs=2,
        type=int
    )
    sp.add_argument(
        '-w', '--path_width',
        action='store',
        default=0.2,
        help=(
            'The width of the paths in the maze as a proportion of the'
            'width of the fill.'
        ),
        type=int
    )
    sp.add_argument(
        '-i', '--inset',
        action='store',
        default=(1, 1),
        help='The depth of the wall on the top and left sides.',
        type=int
    )
    sp.add_argument(
        '-o', '--origin',
        action='store',
        default='tl',
        help='The location to start drawing the path.',
        type=str
    )
    sp.add_argument(
        '-m', '--min',
        action='store',
        default=0x00,
        help='The minimum brightness of the noise.',
        type=int
    )
    sp.add_argument(
        '-M', '--max',
        action='store',
        default=0xff,
        help='The maximum brightness of the noise.',
        type=int
    )
    sp.add_argument(
        '-r', '--repeats',
        action='store',
        default=1,
        help='The how often the values can be repeated within the noise.',
        type=int
    )


def add_noise_arguments(sp: ap.ArgumentParser) -> None:
    """Add noise arguments to a subparser.
    
    :param sp: A subparser that accepts noise arguments.
    :return: None.
    :rtype: NoneType
    """
    sp.add_argument(
        '-s', '--seed',
        action='store',
        default=None,
        help='A seed for the random number generation.'
    )


def add_octave_arguments(
    sp: ap.ArgumentParser,
    d: OctaveNoiseDefaults = OctaveNoiseDefaults()
) -> None:
    """Add octave noise arguments to a subparser.
    
    :param sp: A subparser that accepts noise arguments.
    :param d: Default settings for the noise arguments.
    :return: None.
    :rtype: NoneType
    """
    sp.add_argument(
        '-o', '--octaves',
        action='store',
        default=d.octaves,
        help='The number of octaves.',
        type=int
    )
    sp.add_argument(
        '-p', '--persistence',
        action='store',
        default=d.persistence,
        help='The persistence of octaves.',
        type=int
    )
    sp.add_argument(
        '-a', '--amplitude',
        action='store',
        default=d.amplitude,
        help='The amplitude of octaves.',
        type=int
    )
    sp.add_argument(
        '-f', '--frequency',
        action='store',
        default=d.frequency,
        help='The frequency of octaves.',
        type=int
    )


def add_unitnoise_arguments(sp: ap.ArgumentParser) -> None:
    """Add unit noise arguments to a subparser.
    
    :param sp: A subparser that accepts noise arguments.
    :return: None.
    :rtype: NoneType
    """
    sp.add_argument(
        '-u', '--unit',
        action='store',
        default=(1024, 1024),
        help='The size of a unit within the noise.',
        nargs=2,
        type=int
    )
    sp.add_argument(
        '-m', '--min',
        action='store',
        default=0x00,
        help='The minimum brightness of the noise.',
        type=int
    )
    sp.add_argument(
        '-M', '--max',
        action='store',
        default=0xff,
        help='The maximum brightness of the noise.',
        type=int
    )
    sp.add_argument(
        '-r', '--repeats',
        action='store',
        default=1,
        help='The how often the values can be repeated within the noise.',
        type=int
    )


def add_worley_arguments(sp: ap.ArgumentParser) -> None:
    """Add worley noise arguments to a subparser.
    
    :param sp: A subparser that accepts noise arguments.
    :return: None.
    :rtype: NoneType
    """
    sp.add_argument(
        '-O', '--origin',
        action='store',
        default=(0, 0),
        help='The point within the volume to start taking noise.',
        nargs=2,
        type=int
    )
    sp.add_argument(
        '-P', '--points',
        action='store',
        default=16,
        help='The number of cells in the noise.',
        nargs=2,
        type=int
    )
    sp.add_argument(
        '-v', '--volume',
        action='store',
        default=(-1, -1),
        help='The total volume of noise to generate.',
        nargs=2,
        type=int
    )


# Mainline.
if __name__ == '__main__':
    args = parse_invocation()
    a = args.func(args)
    write(args.path, a)
