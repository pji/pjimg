"""
test_decorators
~~~~~~~~~~~~~~~

Unit tests for :mod:`pjimg.filters.decorators`.
"""
import numpy as np

from pjimg.filters import decorators as decor


# Test cases.
def test_will_square():
    """Given an array with the X axis having a different size
    than the Y axis, :func:`will_square` should make the size
    of those axes the same before passing the array to the
    function. Then slice the result to be the original size of
    the given array.
    """
    @decor.will_square
    def spam(a):
        a[0] = np.rot90(a[0], 1, (0, 1))
        return a

    assert (spam(np.array([[
        [1.0, 1.0, 1.0, 1.0, 1.0,],
        [1.0, 1.0, 1.0, 1.0, 1.0,],
        [1.0, 1.0, 1.0, 1.0, 1.0,],
    ]], dtype=float)) == np.array([[
        [0.0, 1.0, 1.0, 1.0, 0.0,],
        [0.0, 1.0, 1.0, 1.0, 0.0,],
        [0.0, 1.0, 1.0, 1.0, 0.0,],
    ]], dtype=float)).all()
