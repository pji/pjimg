import numpy as np

import pjimg.blends as blnd
import pjimg.eases as ease
import pjimg.filters as filt
import pjimg.sources as srcs
import pjimg.util as util
from pjimg.imgio import write


NAME = '/Users/pji/Pictures/distortions/pd09/pd09.jpg'
TITLE = 'motion.distortion.0j.pji'

Z, Y, X = 0, 1, 2
HEIGHT = 720 // 2
fsize = (240, 16 * HEIGHT // 9, 16 * HEIGHT // 9)
size = (fsize[Z] // 2, fsize[Y], fsize[X])
locs = (
    (0, 0, 0),
    (fsize[Z] // 2, 0, 0),
)


img = np.zeros(fsize, dtype=float)
units = (HEIGHT // 80, HEIGHT // 9, HEIGHT // 9)
v_curtains = srcs.BorktaveCosineCurtains(
    octaves=4,
    persistence=8,
    amplitude=8,
    frequency=-1.5,
    unit=units,
    seed=TITLE + 'v'
)
v2_curtains = srcs.BorktaveCosineCurtains(
    octaves=4,
    persistence=8,
    amplitude=8,
    frequency=-1.5,
    unit=units,
    seed=TITLE + 'v2'
)
v3_curtains = srcs.BorktaveCosineCurtains(
    octaves=4,
    persistence=8,
    amplitude=8,
    frequency=-1.5,
    unit=units,
    seed=TITLE + 'v3'
)
h_curtains = srcs.BorktaveCosineCurtains(
    octaves=4,
    persistence=8,
    amplitude=8,
    frequency=-1.5,
    unit=units,
    seed=TITLE + 'h'
)
h2_curtains = srcs.BorktaveCosineCurtains(
    octaves=4,
    persistence=8,
    amplitude=8,
    frequency=-1.5,
    unit=units,
    seed=TITLE + 'h2'
)
h3_curtains = srcs.BorktaveCosineCurtains(
    octaves=4,
    persistence=8,
    amplitude=8,
    frequency=-1.5,
    unit=units,
    seed=TITLE + 'h3'
)
for loc in locs:
    slices = [slice(n, n + s) for n, s in zip(loc, size)]
    img[*slices] = v_curtains.fill(size, loc)
#     img[*slices] = ease.in_out_quint(img[*slices])
#     print(np.max(img))
#     img /= 0.76
    layer = h_curtains.fill(size, loc)
    layer = filt.rotate_90(layer)
#     layer = ease.in_out_quint(layer)
#     print(np.max(layer))
#     layer /= 0.64
    img[*slices] = blnd.difference(img[*slices], layer)
    layer = v2_curtains.fill(size, loc)
    img[*slices] = blnd.difference(img[*slices], layer)
    layer = h2_curtains.fill(size, loc)
    layer = filt.rotate_90(layer)
    img[*slices] = blnd.difference(img[*slices], layer)
    layer = v3_curtains.fill(size, loc)
    img[*slices] = blnd.difference(img[*slices], layer)
    layer = h3_curtains.fill(size, loc)
    layer = filt.rotate_90(layer)
    img[*slices] = blnd.difference(img[*slices], layer)
    img %= 1.0


img = util.crop_array(img, (240, HEIGHT, 16 * HEIGHT // 9))
img = util.resize_array(img, (240, 720, 1280))


write('test.mp4', img)
