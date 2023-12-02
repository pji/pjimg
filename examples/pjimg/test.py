import numpy as np

import pjimg.blends as blend
import pjimg.filters as filt
import pjimg.sources as srcs
from pjimg.imgio import write


def warp(a):
    a[a != 0] = np.log(a[a != 0])
    return a


size = (1, 720, 1280)
# wave = srcs.Waves()
wave = srcs.Waves(unit=1280, angle=45, wavelength=0.5, warp=warp) 
img = wave.fill(size)
layer = wave.fill(size)
layer = filt.filter_flip(layer, axis=-1)
img = blend.difference(img, layer)
layer = img.copy()
layer = filt.filter_flip(layer, axis=-2)
img = blend.difference(img, layer)
img = filt.filter_colorize(img, colorkey='b')

write('test.jpg', img)
