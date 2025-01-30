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
size = (3, 12, 8)
src = srcs.WorleyCell(points=5, volume=None, seed=seed)

result = (src.fill(size) * 0xff).astype(np.uint8)
print_array(result)
