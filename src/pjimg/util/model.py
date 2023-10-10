"""
model
~~~~~

Common types used by :mod:`pjimg`.
"""
from typing import Union

import numpy as np
from numpy.typing import ArrayLike, NDArray


# Exported names.
__all__ = [
    'ArrayLike', 'ImgAry', 'IntAry', 'Loc', 'NumAry', 'Numeric', 'Size',
]


# Basic types.
ImgAry = NDArray[np.float_]
IntAry = NDArray[np.uint8]
Loc = tuple[int, int, int]
Numeric = Union[np.bool_, np.integer, np.inexact]
Size = tuple[int, int, int]

# Compound types.
NumAry = NDArray[Numeric]
