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
size = (1, 9, 9)
# src = srcs.CosineCurtains(
#     unit=(4, 4, 4),
#     seed=seed,
#     device='cpu'
# )
src = srcs.SolvedMaze(
    start='bl', width=0.34, unit=(1, 3, 3), seed='spam'
)
result = (src.fill(size) * 0xff).astype(np.uint8)
print_array(result, depth=depth)
