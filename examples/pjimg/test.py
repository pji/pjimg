import numpy as np

import pjimg.blends as blnd
import pjimg.eases as ease
import pjimg.filters as filt
import pjimg.sources as srcs
import pjimg.util as util
from pjimg.imgio import write


Z, Y, X = 0, 1, 2
HEIGHT = 720
# fsize = (1, HEIGHT, 16 * HEIGHT // 9)
size = (1, HEIGHT, 16 * HEIGHT // 9)
# size = util.get_free_rotation_size_2d(fsize)
# loc = (1, HEIGHT // 6, -(16 / 6) * HEIGHT // 9)
rotate = 2 * np.pi / 12

bg = srcs.Solid(0.0)
img = bg.fill(size)
img = filt.colorize(img, colorkey='a')

vp = HEIGHT / 10
gap = HEIGHT / 110
print(f'vp = {vp}')
print(f'gap = {gap}')

stops = (0, 1, 0.3, 1, 0.75, 0.7, 1, 0)
grad = srcs.Gradient(stops=stops)
heat = grad.fill(size)
tile = srcs.Tile(
    'octagonwithsquares', vp, gap=gap, color=0.5, color_img=heat
)
img = tile.fill(size)

# layer = img.copy()
# layer = filt.rotate_2d(layer, 45)
# img = blnd.multiply(img, layer)
# starts = [(n - f) // 2 for n, f in zip(size, fsize)]
# stops = [s + f for s, f in zip(starts, fsize)]
# print(starts, stops, size, fsize)
# img = img[0:, starts[Y]:stops[Y], starts[X]:stops[X]]
# print(img.shape)
# 
# layer_a = filt.gaussian_blur(img, 40)
# layer_b = filt.gaussian_blur(img, 20)
# layer = blnd.multiply(layer_a, layer_b)
# layer_c = filt.gaussian_blur(img, 10)
# layer = blnd.multiply(layer, layer_c)
# layer_c = filt.gaussian_blur(img, 5)
# layer = blnd.multiply(layer, layer_c)
# layer = ease.in_quad(layer)
# img = blnd.multiply(img, layer)
# 
# shade = filt.colorize(img, colorkey='a')
# shade *= 0.6
# shade += 0.4
# img *= 0.7
# img += 0.3
# img = filt.colorize(img, colorkey='bp')
# img = blnd.multiply(img, shade)
# img = blnd.overlay(img, shade, fade=0.8)

# reg = srcs.Regular(4, 5, color=1, bg_color=0, antialias=True)
# layer = reg.fill(size)
# img = blnd.lighter(img, layer)
# mask = filt.colorize(mask, colorkey='a')
# 
# solid = srcs.Solid(0.5)
# layer = solid.fill(size)
# layer = filt.colorize(layer, colorkey='s')
# img = blnd.replace(img, layer, mask=mask)

# layer = img.copy()
# layer = filt.pinch(layer, -0.75, HEIGHT // 2, (1, 1, 1))
# img = blnd.exclusion(img, layer)

# img = filt.gaussian_blur(img, HEIGHT / 100)
# img = ease.in_quad(img)

# unit = 35 * HEIGHT // 9
# units = (18, unit, unit)
# operlin = srcs.OctavePerlin(
#     octaves=6, unit=units, seed='pd09'
# )
# layer = operlin.fill(size)
# for i in range(4):
#     operlin.seed = f'pd09-{i}'
#     layer = blnd.difference(layer, operlin.fill(size))
# layer = filt.contrast(layer)
# layer = ease.in_out_sin(layer)
# layer /= 2
# layer += 0.2
# 
# layer = filt.colorize(layer, colorkey='g')
# mask = filt.inverse(mask)
# img = blnd.replace(img, layer, mask=mask)

write('test.jpg', img)
