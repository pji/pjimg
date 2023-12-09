import numpy as np

import pjimg.blends as blend
import pjimg.eases as ease
import pjimg.filters as filt
import pjimg.sources as srcs
from pjimg.imgio import write


def warp(a):
    a[a != 0] = np.log(a[a != 0])
    return a


size = (1, 720, 1280)
# wave = srcs.Waves()
wave = srcs.Waves(
    unit=1280,
    wavelength=1.1,
    angle=30,
    warp=warp
    # radial=True
) 
img = wave.fill(size)
layer = wave.fill(size, (0, 0, -210))
layer = filt.flip(layer, axis=-1)
img = blend.difference(img, layer)

layer = wave.fill(size, (0, 0, -80))
layer = filt.flip(layer, axis=-2)
img = blend.difference(img, layer)

# img *= 0.8
# img += 0.2
gl = img.copy()
hl = img.copy()
sd = img.copy()

# perlin = srcs.OctavePerlin()
# layer = perlin.fill(size)
# 
# img = filt.posterize(img, 40)
# img[img > 0.075] = 1.0
# img[img < 0.05] = 0.0
# layer[img == 1.0] = 1.0
# layer[img == 0.0] = 0.0
# layer = filt.posterize(layer, 2)
# img = blend.screen(img, layer)
# img = filt.gaussian_blur(img, 10)
img = filt.colorize(img, colorkey='cp')

sd *= 0.8
sd += 0.2
sd = filt.colorize(sd, colorkey='a')
img = blend.multiply(img, sd)

# gl = ease.in_quad(gl)
# gl = filt.gaussian_blur(gl, 10)
# gl = filt.colorize(gl, colorkey='a')
# 
# hl = ease.in_quint(hl)
# hl = filt.colorize(hl, colorkey='a')
# img = blend.screen(img, hl)

write('test.jpg', img)
