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
layer = filt.filter_flip(layer, axis=-1)
img = blend.difference(img, layer)

# layer = img.copy()
# layer = filt.filter_flip(layer, axis=-2)
# img = blend.difference(img, layer)

layer = wave.fill(size, (0, 0, -80))
layer = filt.filter_flip(layer, axis=-2)
img = blend.difference(img, layer)

# layer = wave.fill(size, (0, 0, -40))
# layer = filt.filter_flip(layer, axis=-1)
# layer = filt.filter_flip(layer, axis=-2)
# img = blend.difference(img, layer)

img *= 0.8
img += 0.2
layer = img.copy()
hl = img.copy()
sd = img.copy()
# img = ease.ease_out_circ(img)

# img = ease.ease_out_quad(img)
img = filt.filter_colorize(img, colorkey='cp')
# img[img < 0.25] = filt.filter_colorize(img[img < 0.25], colorkey='b')

# sd = filt.filter_inverse(sd)
# # sd = ease.ease_in_quint(sd)
# # sd = ease.ease_in_quint(sd)
# # sd = ease.ease_in_quad(sd)
# # sd = ease.ease_in_quint(sd)
# # sd = filt.filter_gaussian_blur(sd, 0.5)
# # sd = filt.filter_gaussian_blur(sd, 0.5)
sd = filt.filter_colorize(sd, colorkey='a')
# sd = ease.ease_in_sin(sd)
# sd *= 0.75
# sd += 0.25
img = blend.multiply(img, sd)

layer = ease.ease_in_quad(layer)
layer = filt.filter_gaussian_blur(layer, 10)
layer = filt.filter_colorize(layer, colorkey='a')
# img = blend.screen(img, layer)

hl = ease.ease_in_quint(hl)
hl = filt.filter_colorize(hl, colorkey='a')
img = blend.screen(img, hl)

write('test.jpg', img)
# write('test.mp4', img)
