"""
mazer
=====

Create a printable maze using :mod:`pjimg`.
"""
from argparse import ArgumentParser, Namespace
from datetime import date
from pathlib import Path
from typing import Union

from pjimg.imgblend import difference, lighter
from pjimg.imggen import Box, Maze, Seed, SolvedMaze, Text
from pjimg.imgio import write
from pjimg.util import ImgAry, Loc, Size, X_, Y_


# Create the image data.
def make_maze(
    origin: Loc = (0, 0, 0),
    path: str = 'maze.png',
    seed: Seed = None,
    size: Size = (1, 720, 560),
    solve: bool = False,
    unit: int = 20,
) -> ImgAry:
    """Build a printable maze.
    
    :param origin: (Optional.) Where in the image to start building the
        path through the maze.
    :param path: (Optional.) Where to save the image file.
    :param seed: (Optional.) The value to use to seed the random number
        generator used in making the maze.
    :param size: (Optional.) The Z, Y, and X pixel dimensions of the maze.
    :param solve: (Optional.) Whether to add the solution to the maze.
    :param unit: (Optional.) The size in pixels of each unit in the grid
        used to build the maze. It's the width of the paths in the maze
        plus half of the width of the walls on either side.
    :return: None.
    :rtype: NoneType
    """
    # Calculate the remaining configuration.
    units = (1, unit, unit)
    exit_size = (1, int(unit * 0.8), unit)
    
    # Create maze interior.
    maze = Maze(width=0.4, origin=origin, unit=units, seed=seed)
    img = maze.fill(size)
    
    # Add the maze entrance.
    entrance = Box(
        origin=(0, int(unit * 0.6), 0),
        dimensions=exit_size,
        color=1.0
    )
    img = lighter(img, entrance.fill(size))
    
    # Add the maze exit.
    exit = Box(
        origin=(0, int(size[Y_] - unit * 1.4), size[X_] - unit),
        dimensions=exit_size,
        color=1.0
    )
    img = lighter(img, exit.fill(size))
    
    # Add the solution.
    if solve:
        solution = SolvedMaze(
            width=0.1,
            origin=origin,
            unit=units,
            seed=seed
        )
        fill = solution.fill(size)
        fill *= 0.5
        img = difference(img, fill)
    
    # Add the title of the maze.
    title = Text(
        seed,
        size=int(unit * 0.5),
        origin=(int(unit * 0.6), 1),
        font='Helvetica',
        face=1
    )
    img = lighter(img, title.fill(size))
    
    # Return the image data.
    return img


# Save the image data.
def save_maze_file(img: ImgAry, path: Union[Path, str]) -> None:
    """Save the image data as a file.
    
    :param img: The array of image data.
    :param path: Where to save the image file.
    :return: None.
    :rtype: NoneType
    """
    # Avoid clobbering exiting files.
    path = Path(path)
    if path.exists():
        print('Error: Cannot overwrite existing file or directory.')
    
    # Save the file.
    else:
        write(path, img)


# Utility functions.
def get_today(delim: str = '.') -> str:
    """Get today's date.
    
    :param delim: Delimiter for the date stamp.
    :return: Today's date as a :class:`str`.
    :rtype: str
    """
    today = date.today()
    year, month, day, *_ = today.timetuple()
    return f'{year}{delim}{month:02d}{delim}{day:02d}'


def parse_cli() -> Namespace:
    """Parse the command line arguments used to invoke :mod:`mazer`.
    
    :return: The arguments as a :class:`argparse.Namespace`.
    :rtype: argparse.Namespace
    """
    p = ArgumentParser(
        prog='MAZER',
        description='Create a printable maze.'
    )
    p.add_argument(
        '-d', '--dimensions',
        action='store',
        default=(560, 720),
        help='Width and height of the maze image.',
        nargs=2,
        type=int
    )
    p.add_argument(
        '-O', '--out',
        action='store',
        default='',
        help='Where to save the file.',
        type=str
    )
    p.add_argument(
        '-o', '--origin',
        action='store',
        default='m',
        help='Where in the maze the generation should start.',
        type=str
    )
    p.add_argument(
        '-S', '--solve',
        action='store_true',
        help='Include the solution to the maze.'
    )
    p.add_argument(
        '-s', '--seed',
        action='store',
        default=get_today(),
        help='Seed used to generate the maze.',
        type=str
    )
    p.add_argument(
        '-u', '--unit',
        action='store',
        default=20,
        help='The unit size for the maze grid.',
        type=int
    )
    return p.parse_args()


# Mainline.
if __name__ == '__main__':
    # Configure the maze generation from the command line arguments.
    args = parse_cli()
    path = args.out
    if not path and not args.solve:
        path = f'{args.seed}.png'
    if not path:
        path = f'{args.seed}_solved.png'
    width, height = args.dimensions
        
    # Generate the maze.
    img = make_maze(
        origin=args.origin,
        path=path,
        seed=args.seed,
        size=(1, height, width),
        solve=args.solve,
        unit=args.unit
    )
    
    # Save the image.
    save_maze_file(img, path)
