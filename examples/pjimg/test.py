import numpy as np

import pjimg.blends as blend
import pjimg.eases as ease
import pjimg.filters as filt
import pjimg.sources as srcs
from pjimg.imgio import write


HEIGHT = 720
size = (1, HEIGHT, 16 * HEIGHT // 9)
loc = (1, HEIGHT // 6, -(16 / 6) * HEIGHT // 9)
rotate = 2 * np.pi / 12
regular = srcs.Regular(6, HEIGHT // 6, rotate=rotate, color=0.5)
img = regular.fill(size, loc)

write('test.jpg', img)
