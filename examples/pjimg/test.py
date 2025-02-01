from concurrent import futures
from datetime import datetime

import numpy as np

import pjimg.blends as blnd
import pjimg.eases as ease
import pjimg.filters as filt
import pjimg.sources as srcs
import pjimg.util as util
from pjimg.imgio import write


NAME = 'pp03.jpg'
TITLE = 'protopattern.03.pji'

Z, Y, X = 0, 1, 2
t0 = datetime.now()
WIDTH = 720
UNITS = (1, 320, 320)
size = (1, 16 * WIDTH // 9, WIDTH)
SEED = TITLE
COLORKEY = 'C'

# Build image.
src = srcs.OctaveMaze(6, unit=UNITS, seed=SEED)
img = src.fill(size)
# img = filt.inverse(img)
img = filt.colorize(img, COLORKEY)

# Add the title.
HEIGHT = size[1]
font_size = WIDTH // 40
text = srcs.Text(
    text=TITLE,
    font='Menlo',
    size=font_size,
    face=1,
    fill_color=1.0,
    bg_color=0.5
)
loc = (0, WIDTH // 40, HEIGHT - font_size - WIDTH // 60)
layer = text.fill(size, loc)
layer = filt.colorize(layer, colorkey='a')
img = blnd.overlay(img, layer)
img = blnd.overlay(img, layer)

# Save image.
print(f'Postprocessing complete at {datetime.now() - t0}.')
write(NAME, img)
