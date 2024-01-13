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
FRAMES = 1
HEIGHT = 720
size = (FRAMES, HEIGHT, 16 * HEIGHT // 9)

grad = srcs.Gradient()
img = grad.fill(size)
ease.in_out_bounce(img)

write('test.jpg', img)
