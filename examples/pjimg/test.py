from datetime import datetime

import numpy as np

import pjimg.blends as blnd
import pjimg.eases as ease
import pjimg.filters as filt
import pjimg.sources as srcs
import pjimg.util as util
from pjimg.imgio import write


NAME = '/Users/pji/Pictures/distortions/pd09/pd09.jpg'
TITLE = 'motion.distortion.0j.mp4'

Z, Y, X = 0, 1, 2
FRAMES = 120
HEIGHT = 720
size = (FRAMES, HEIGHT, 16 * HEIGHT // 9)
t0 = datetime.now()

small = [n // 8 for n in size]
unit = small[Y] * 2
operlin = srcs.OctavePerlin(
    octaves=4, unit=(FRAMES, unit, unit), seed=TITLE + 'a'
)
print('Layer 1 fill.')
img = operlin.fill(small)
# print('Layer 1 ease.')
# img = ease.in_out_bounce(img)
print(f'size = {img.shape}')
print(f'max = {np.max(img)}')
print(f'min = {np.min(img)}')
print(f'duration = {datetime.now() - t0}')

print('Image resize.')
img = util.resize_array(img, size, util.ndcerp)
print(f'size = {img.shape}')
print(f'max = {np.max(img)}')
print(f'min = {np.min(img)}')
print(f'duration = {datetime.now() - t0}')

write('test_full.mp4', img)
