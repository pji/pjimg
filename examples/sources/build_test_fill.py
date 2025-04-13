"""
build_test_fill
~~~~~~~~~~~~~~~

Create the unit test fill data for a unit test for a
:class:`pjimg.sources.model.Source` class.
"""
import numpy as np

from pjimg.util.debug import print_array
import pjimg.sources as srcs


depth = 2
seed = 'spam'
size = (3, 8, 8)
src = srcs.UnitNoiseTorch(
    unit=(4, 4, 4),
    seed=seed
)

result = (src.fill(size) * 0xff).astype(np.uint8)
print_array(result, depth=depth)
