import numpy as np

import pjimg.blends as blnd
import pjimg.eases as ease
import pjimg.filters as filt
import pjimg.sources as srcs
from pjimg.imgio import write


HEIGHT = 720
size = (1, HEIGHT, 16 * HEIGHT // 9)
# loc = (1, HEIGHT // 6, -(16 / 6) * HEIGHT // 9)
rotate = 2 * np.pi / 12

bg = srcs.Solid(0.0)
img = bg.fill(size)
img = filt.colorize(img, colorkey='a')

stops = (0, 1, 0.3, 1, 0.75, 0.7, 1, 0)
grad = srcs.Gradient(stops=stops)
heat = grad.fill(size)
tile = srcs.Tile(
    'hexagon', HEIGHT / 20, gap=HEIGHT // 100, color=0.5, rotation=np.pi / 12
)
img = tile.fill(size)
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
