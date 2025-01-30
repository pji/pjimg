from concurrent import futures
from datetime import datetime

import numpy as np

import pjimg.blends as blnd
import pjimg.eases as ease
import pjimg.filters as filt
import pjimg.sources as srcs
import pjimg.util as util
from pjimg.imgio import write


NAME = 'pp01.jpg'
TITLE = 'protopattern.01.jpg'

Z, Y, X = 0, 1, 2
t0 = datetime.now()
HEIGHT = 720
UNITS = (1, 20, 20)
size = (1, 16 * HEIGHT // 9, HEIGHT)
SEED = 'worley'
COLORKEY = 'e'

# Build image.
src = srcs.Worley(points=100, seed=SEED)
img = src.fill(size)
img = filt.colorize(img, COLORKEY)


# Save image.
print(f'Postprocessing complete at {datetime.now() - t0}.')
write(NAME, img)
