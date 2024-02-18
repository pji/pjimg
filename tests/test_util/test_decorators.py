"""
test_decorators
~~~~~~~~~~~~~~~

Unit tests for :mod:`pjimg.util.decorators`.
"""
import numpy as np

import pjimg.util.decorators as d


# Test cases.
def test_preserves_type():
    """Given a function that will change the data type of an array,
    :func:`preserve_type` should ensure the returned array has the
    same data type as the original array.
    """
    @d.preserves_type
    def change_type(a):
        return a.astype(int)
    
    a = np.array([0.0, 0.5, 1.0,], dtype=float)
    assert change_type(a).dtype is np.dtype('float')
