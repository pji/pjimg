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
size = (1, HEIGHT, 16 * HEIGHT // 9)
SEED = 'spam'
COLORKEY0 = 'b'
COLORKEY1 = 'p'

# Build image.
# Base sources.
a_src = srcs.Solid(0.0)
b_src = srcs.Solid(1.0)
mask_src = srcs.WorleyCell(points=8, seed=SEED)

# Base data.
a = a_src.fill(size)
b = b_src.fill(size)
mask = mask_src.fill(size)

# Layer 0.
img = blnd.replace(a, b, mask=mask)
img = filt.colorize(img, COLORKEY0)

# Layer 1.
mask_inverse = filt.inverse(mask)
layer = blnd.replace(a, b, mask=mask_inverse)
layer = filt.colorize(layer, COLORKEY1)
mask_inverse = filt.colorize(mask_inverse, 'w')
img = blnd.replace(img, layer, mask=mask_inverse)

# Save image.
print(f'Postprocessing complete at {datetime.now() - t0}.')
write(NAME, img)
