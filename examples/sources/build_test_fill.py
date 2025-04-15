"""
build_test_fill
~~~~~~~~~~~~~~~

Create the unit test fill data for a unit test for a
:class:`pjimg.sources.model.Source` class.
"""
import numpy as np

from pjimg.util.debug import print_array
import pjimg.sources as srcs


depth = 1
seed = 'spam'
size = (3, 12, 10)
src = srcs.OctavePerlin(
    unit=(4, 4, 4),
    seed=seed,
    device='cpu'
)

result = (src.fill(size) * 0xff).astype(np.uint8)
print_array(result, depth=depth)
