"""
modister
========

Generate a motion distortion.
"""
import operator as op
import subprocess
from argparse import ArgumentParser, Namespace
from pathlib import Path
from queue import Queue
from random import getrandbits
from sys import stdout
from threading import Thread
from time import sleep
from traceback import print_exc
from typing import Sequence, Union

import thurible as thur
import thurible.messages as tmsg
import thurible.progress as rprg

import pjimg.eases as ease
import pjimg.filters as filt
import pjimg.imgio as imgio
import pjimg.sources as srcs
from pjimg.blends import difference, overlay
from pjimg.util import ImgAry, Loc, Size, X, Y, Z


# Build the image.
def add_grain(
    img: ImgAry,
    seeds: Sequence[str],
    size: Size,
    fade: float = 0.1
) -> ImgAry:
    """Add graininess to the image to prevent banding in the gradients.
    
    :param img: The image data to add grain to.
    :param seeds: The list of seeds used to generate the grain.
    :param size: The shape of the array of image data.
    :param fade: How much the grain should affect the image data.
    :return: Image data in a :class:`numpy.ndarray`.
    :rtype: numpy.ndarray
    """
    seed = ''.join(seed for seed in seeds)
    src = srcs.Noise(seed=seed)
    grain = src.fill(size)
    return overlay(img, grain, fade=0.1)


def build_curtain(
    units: Size,
    size: Size,
    seed: str,
    img: Union[ImgAry, None],
    rotate: bool = False,
    loc: Loc = (0, 0, 0)
) -> ImgAry:
    """Build one curtain of the motion distortion."""
    src = srcs.BorktaveCosineCurtains(
        octaves=4,
        frequency=1.5,
        unit=units,
        seed=seed
    )
    if rotate:
        size = (size[Z], size[X], size[Y])
    new = src.fill(size)
    if rotate:
        new = filt.filter_rotate_90(new)
    if img is not None:
        new = difference(img, new)
    return new


def build_modist(
    size: Size = (120, 720, 1280),
    unit: int = 180,
    speed: int = 12,
    seeds: Sequence[str] = ('0', '1'),
    loc: Loc = (0, 0, 0),
    q_to: Union[None, Queue] = None
) -> ImgAry:
    """Layer curtains to create a motion distortion."""
    img = None
    for i, seed in enumerate(seeds):
        if q_to is not None:
            q_to.put(NoTick(f'Curtain {i} started.'))
        rotate = False
        if i % 2 == 0:
            rotate = True
        unit_mod = int((unit / 8) * i)
        speed_mod = (speed // 8) * i
        units = (speed + speed_mod, unit + unit_mod, unit + unit_mod)
        img = build_curtain(units, size, seed, img, rotate, loc)
        if q_to is not None:
            q_to.put(Tick(f'Curtain {i} blended.'))
    return img


def build_seeds(
    seeds: Union[None, Sequence[str]],
    num: int,
    path: Union[Path, str],
    length: int = 32
) -> list[str]:
    """build the seed values for the curtains."""
    if path:
        seeds = read_seeds(path)
    if not seeds:
        seeds = [getrandbits(length) for _ in range(num)]
        seeds = [str(seed) for seed in seeds]
    return seeds


def build_slices(
    size: Size,
    unit: int,
    speed: int,
    seeds: Union[None, Sequence[str]],
    q_to: Queue,
    colorkey: str,
    path: Union[Path, str],
    framerate: int,
    loc: Loc = (0, 0, 0)
) -> None:
    """Build the modist in slices."""
    path = Path(path)
    frames = size[Z]
    slice_frames = 120
    
    slice_size = (slice_frames, size[Y], size[X])
    slice_ids = [sid for sid in range(0, frames, slice_frames)]
    units = [speed, unit, unit]
    
    q_to.put(rprg.NoTick(f'Making sources...'))
    sources = []
    for seed in seeds:
        source = srcs.OctavePerlin(
            octaves=3,
            persistence=4.0,
            amplitude=24.0,
            frequency=4.0,
            unit=units,
            seed=seed
        )
        sources.append(source)
        q_to.put(rprg.NoTick(f'Made {seed} source...'))
    q_to.put(rprg.NoTick(f'Sources made...'))
    
    for frame_loc in slice_ids:
        slice_loc = (loc[Z] + frame_loc, loc[Y], loc[X])
        
        slice_path = path.parent / f'{path.stem}_{frame_loc}{path.suffix}'
        img = render_modist(sources, slice_size, slice_loc, q_to)
        img = post_process_modist(
            img,
            seeds,
            colorkey,
            q_to,
            slice_size
        )
        save_modist(slice_path, img, framerate)
    concat_slices(path, slice_ids, q_to)


def concat_slices(
    path: Path,
    slice_ids: Sequence[int],
    q_to: Queue,
    out: int = subprocess.DEVNULL
) -> None:
    """Use ffmpeg to put the slices of video together."""
    filelist = path.parent / f'{path.stem}_files.txt'
    for sid in slice_ids:
        slice_file = path.parent / f'{path.stem}_{sid}{path.suffix}'
        with open(filelist, 'a') as fh:
            fh.write(f"file '{slice_file}'\n")
    
    q_to.put(rprg.NoTick('Start TS file concatenation.'))
    subprocess.call([
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0',
        '-i', f'{filelist}',
        '-c', 'copy',
        f'{path}'
    ], stdout=out, stderr=out)
    q_to.put(rprg.Tick('Finished TS file concatenation.'))


def post_process_modist(
    img: ImgAry,
    seeds: Sequence[str],
    colorkey: str,
    q_to: Queue,
    size: Size
) -> ImgAry:
    """Perform post-processing steps."""
    steps = [
        [ease.ease_in_out_quad, [img,]],
        [add_grain, [img, seeds, size]],
        [op.mod, [img, 1.0]],
        [filt.filter_colorize, [img, colorkey]],
    ]
    for step in steps:
        fn, args = step
        q_to.put(rprg.NoTick(f'{fn.__name__} started.' ))
        img = fn(*args)
        q_to.put(rprg.NoTick(f'{fn.__name__} complete.' ))
    return img


def read_seeds(path: Path):
    """Read the seeds from a file."""
    if not path.exists():
        raise FileNotFoundError(f'Does not exist: {path}.')
    with open(path) as fh:
        lines = fh.readlines()
    return [line[:-1] for line in lines]


def render_layer(
    source: srcs.Source,
    size: Size,
    loc: Loc,
    old: Union[ImgAry, None] = None
) -> ImgAry:
    """Render one layer of the image."""
    img = source.fill(size, loc)
    if old is not None:
        img = difference(old, img)
    return img


def render_modist(
    sources: Sequence[srcs.Source],
    size: Size,
    loc: Loc,
    q_to: Queue
) -> ImgAry:
    """Render the motion distortion."""
    img: Union[ImgAry, None] = None
    for i, source in enumerate(sources):
        q_to.put(rprg.Tick(f'Layer {i} started.'))
        img = render_layer(source, size, loc, img)
        q_to.put(rprg.Tick(f'Layer {i} is blended.'))
    return img


def save_modist(path: Path, img: ImgAry, framerate: int = 12) -> ImgAry:
    """Save the motion distortion to a file."""
    imgio.write_video(path, img, framerate)


def save_seeds(path: Path, seeds: Sequence[str]) -> None:
    """Save the seeds to a file."""
    seed_path = path.parent / f'{path.stem}_seeds.txt'
    with open(seed_path, 'w') as fh:
        for seed in seeds:
            fh.write(f'{seed}\n')
    print(f'Seeds saved as {seed_path}.')


# Manage the UI.
def build_progress_bar(steps: int) -> tuple[Thread, Queue, Queue]:
    """Build the progress bar UI."""
    q_to, q_from = thur.get_queues()
    progress = thur.Progress(
        steps=steps,
        max_messages=6,
        messages=['Waiting',],
        timestamp=True,
        bar_bg='bright_black',
        frame_type='double',
        title_text='modister',
        title_frame=True,
        panel_relative_height=0.3,
        panel_relative_width=0.9,
        content_relative_width=0.9
    )
    q_to.put(tmsg.Store('progress', progress))
    q_to.put(tmsg.Show('progress'))
    T = Thread(target=thur.queued_manager, args=(q_to, q_from))
    T.start()
    return T, q_to, q_from


def parse_cli() -> Namespace:
    """Parse the command that invoked the script."""
    p = ArgumentParser(
        prog='modister',
        description='Create a motion distortion.'
    )
    p.add_argument(
        'path',
        help='The path to save the modist.',
        type=Path
    )
    p.add_argument(
        '-c', '--colorkey',
        default='a',
        help='The color used for the modist.',
        type=str
    )
    p.add_argument(
        '-d', '--duration',
        default='10.0',
        help='The duration of the modist in seconds.',
        type=float
    )
    p.add_argument(
        '-f', '--framerate',
        default='12',
        help='The framerate of the modist in frames/second.',
        type=int
    )
    p.add_argument(
        '-H', '--height',
        default='720',
        help='The height of the modist in pixels.',
        type=int
    )
    p.add_argument(
        '-L', '--load_seeds',
        help='A file of seeds for the modist.',
        type=Path
    )
    p.add_argument(
        '-p', '--speed',
        default=12,
        help='How quickly things change in the modist.',
        type=int
    )
    p.add_argument(
        '-s', '--seeds',
        help='List of the generation seeds.',
        nargs='*',
        type=str
    )
    p.add_argument(
        '-S', '--num_seeds',
        default=6,
        help='The number of generation seeds to create.',
        type=int
    )
    p.add_argument(
        '-W', '--width',
        default=1280,
        help='The width of the modist in pixels.',
        type=int
    )
    return p.parse_args()


# Mainline.
if __name__ == '__main__':
    args = parse_cli()
    seeds = build_seeds(args.seeds, args.num_seeds, args.load_seeds)
    framerate = args.framerate
    frames = int(framerate * args.duration)
    size = (frames, args.height, args.width)
    speed = args.speed * 48
    unit = args.width * 4
    steps = (frames // 120) * (len(seeds) * 3 + 1) + 7
    
    try:
        T, q_to, q_from = build_progress_bar(steps)
        img = build_slices(
            size=size,
            unit=unit,
            speed=speed,
            seeds=seeds,
            q_to=q_to,
            colorkey=args.colorkey,
            path=args.path,
            framerate=framerate
        )
        q_to.put(rprg.Tick('Modist saved.'))
        q_to.put(tmsg.End(f'Modist saved as {args.path}.'))
    
    except Exception as ex:
        q_to.put(tmsg.End(f'{type(ex).__name__}\n{ex}'))
        sleep(0.5)
        raise ex
    
    except KeyboardInterrupt as ex:
        q_to.put(tmsg.End(f'{type(ex).__name__}'))
        sleep(0.5)
        raise ex
    
    T.join(5)
    save_seeds(args.path, seeds)
