"""
patdister
~~~~~~~~~

Create a pattern distortion.
"""
from argparse import ArgumentParser, Namespace
from pathlib import Path

import pjimg.blends as blend
import pjimg.eases as ease
import pjimg.filters as filt
import pjimg.sources as srcs
from pjimg.imgio import write
from pjimg.util import X, Y, Z


# User interface.
def parse_invocation() -> Namespace:
    """Parse the command line invocation."""
    p = ArgumentParser(
        prog='patdister',
        description='Create a pattern distortion.'
    )
    p.add_argument(
        'path',
        help='The path to save the pattern distortion.',
        type=Path
    )
    p.add_argument(
        '-d', '--dimensions',
        default=(1280, 720),
        help="The width and height of the distortion.",
        nargs=2,
        type=int
    )
    return p.parse_args()


# Mainline.
if __name__ == '__main__':
    args = parse_invocation()
    width, height = args.dimensions
    dim = width if width > height else height
    dim = int(dim * 1.5)
    worksize = (1, dim, dim)
    print('Starting.')

    radius = height // 10
    hexes = srcs.Hexes(radius=radius)
    img = hexes.fill(worksize)
    img = ease.ease_out_quint(img)
    print('Finished layer 1.')
    
    units = (1, height * 4, height * 4)
    perlin = srcs.OctavePerlin(unit=units, seed='spam')
    layer = perlin.fill(worksize)
    layer = ease.ease_in_out_quint(layer)
    img = blend.multiply(img, layer)
    print('Finished layer 2.')
    
    radius = height // 10
    hexes = srcs.Hexes(radius=radius)
    layer = hexes.fill(worksize)
    layer = ease.ease_out_quint(layer)
    img = blend.overlay(img, layer)
    print('Finished layer 3.')
    
    img = filt.filter_inverse(img)
    img_copy = img.copy()
    layer = img_copy
    layer = filt.filter_polar_to_linear(layer)
    layer = filt.filter_motion_blur(layer, amount=width // 8, axis=-1)
    layer = filt.filter_linear_to_polar(layer)
    # img = blend.screen(img, layer, fade=0.6)
    img = layer
    print('Finished layer 4.')
        
    layer = img_copy
    layer = filt.filter_polar_to_linear(layer)
    layer = filt.filter_motion_blur(layer, amount=width // 4, axis=-1)
    img = blend.screen(img, layer, fade=0.6)
    layer = filt.filter_linear_to_polar(layer)
    print('Finished layer 5.')
        
    layer = img_copy
    layer = filt.filter_polar_to_linear(layer)
    layer = filt.filter_motion_blur(layer, amount=width // 16, axis=-1)
    layer = filt.filter_linear_to_polar(layer)
    img = blend.screen(img, layer, fade=0.6)
    print('Finished layer 6.')
        
    layer = img_copy
    layer = filt.filter_polar_to_linear(layer)
    layer = filt.filter_motion_blur(layer, amount=width // 32, axis=-1)
    layer = filt.filter_linear_to_polar(layer)
    img = blend.screen(img, layer, fade=0.6)
    print('Finished layer 6.')
        
    layer = img_copy
    layer = filt.filter_polar_to_linear(layer)
    layer = filt.filter_motion_blur(layer, amount=width // 2, axis=-1)
    layer = filt.filter_linear_to_polar(layer)
    img = blend.screen(img, layer, fade=0.6)
    print('Finished layer 7.')
        
    size = (1, height, width)
    starts = [(w - s) // 2 for w, s in zip(worksize, size)]
    stops = [start + s for start, s in zip(starts, size)]
    img = img[:, starts[-2]:stops[-2], starts[-1]:stops[-1]]
    print('Finished post processing.')
    
    img = ease.ease_in_quint(img)

    write(args.path, img)
    print('Saved.')
