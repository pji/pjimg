"""
build_test_fill
~~~~~~~~~~~~~~~

Create the unit test fill data for a unit test for a
:class:`pjimg.sources.model.Source` class.
"""
import numpy as np

from pjimg.util.debug import print_array
import pjimg.sources as srcs


seed = 'spam'
size = (1, 20, 10)
src = srcs.OctaveMaze(
    octaves=4,
    persistence=2,
    amplitude=2,
    frequency=3,
    unit=(1, 10, 10),
    width=0.34,
    seed=seed
)

result = (src.fill(size) * 0xff).astype(np.uint8)
print_array(result)
